# coding=utf-8
import os
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import pymysql
import streamlit as st
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")


# ---------- Secrets (read at runtime) ----------
# Ensure you have [db_wms] and [db_cbs] set in Streamlit secrets (Advanced settings)
def _get_db_conf(section: str):
    conf = st.secrets.get(section)
    if conf is None:
        raise RuntimeError(f"Secrets section '{section}' not found. Please set it in Streamlit Advanced settings.")
    return conf

@st.cache_data(ttl=60, show_spinner=False)
def get_scf_appointment() -> pd.DataFrame:
    """
    Fetch large-mawb dataset from CBS (cached).
    """
    db_cbs = _get_db_conf("db_wms")
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs.get("port", 3306)),
        user=db_cbs["user"],
        password=db_cbs["password"],
        database=db_cbs["database"],
        charset="utf8"
    )

    sql_ = r"""
    SELECT * FROM (
	SELECT w.warehouse_code AS warehouseCode,pl.pallet_no AS containerNo, IF(pl.`STATUS`= 0,'inbound','loaded') AS conStatus,pl.channel AS channel, pl.customs_status AS cbpStatus,pl.pga_status AS pgaStatus, MIN(m.ata) AS ata,pl.PLT_BOX_NUM AS boxCount,pl.create_time AS createTime  
	FROM ifm_warehouse_pallet pl 
	LEFT JOIN ifm_warehouse_bag b ON b.pallet_id = pl.id AND b.mark = 1
	LEFT JOIN ifm_warehouse_mawb m ON m.id = b.warehouse_mawb_id AND m.mark= 1
	LEFT JOIN sys_warehouse w ON pl.warehouse_id = w.id 
	WHERE pl.mark= 1 AND pl.`STATUS` IN (0,1)
	GROUP BY pl.id
	HAVING count(b.id) > 0
	UNION 
	SELECT w.warehouse_code AS warehouseCode,gl.gayload_no AS containerNo,  IF(gl.`STATUS`= 0,'inbound','loaded') AS conStatus,gl.channel AS channel, gl.customs_status AS cbpStatus,gl.pga_status AS pgaStatus, MIN(m.ata) AS ata,gl.pieces AS boxCount,gl.create_time AS createTime  
	FROM ifm_warehouse_gayload gl 
	LEFT JOIN ifm_warehouse_bag b ON b.gayload_id = gl.id AND b.mark = 1
	LEFT JOIN ifm_warehouse_mawb m ON m.id = b.warehouse_mawb_id AND m.mark= 1
	LEFT JOIN sys_warehouse w ON gl.warehouse_id = w.id 
	WHERE gl.mark= 1 AND gl.`STATUS` IN (0,1)
	GROUP BY gl.id
	HAVING count(b.id) > 0
) AS temp 
WHERE createTime > '2026-01-01 00:00:00' 
ORDER BY warehouseCode,createTime DESC;
    """

    container = pd.read_sql(sql_, conn)  # select来查询数据
    container_ = container[
        (container['cbpStatus'] == 0) &
        (container['pgaStatus'] == 2) &
        (container['warehouseCode'] == 'JFK1')&(container['conStatus']=='inbound')]
    container_info = container_[
        container_['channel'].str.startswith('USPS', na=False) & (container_['channel'].str.len() == 7)]
    container_info['ata']=container_info['ata']-pd.Timedelta(hours=5)
    container_info['scf_type']='1'

    """
    计算KPI预警
    """
    container_info['weekday'] = container_info["ata"].dt.weekday+1

    def compute_base_time(row):
        b = row["weekday"]
        a = row["ata"]

        if b in [6, 7]:  # 周六/周日
            #找到A时间对应周的下周一 00:00
            next_monday = a + pd.offsets.Week(weekday=0)
            base_time = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            result = base_time
        else:
            result = a

        return result

    container_info["base_time"] = container_info.apply(compute_base_time, axis=1)


    """
    读取美国法定假日
    """
    us_holiday = pd.read_excel('USPS美国法定假日.xlsx')
    us_holiday_list=us_holiday['日期'].to_list()
    us_holiday_list = pd.to_datetime(us_holiday_list).date


    def add_3_5_business_days(t):
        # 先加 3 个工作日
        channel_val = t.get('channel', '')
        if channel_val=="USPSGRR":
            # 1️⃣ 先加 3 个工作日
            t['base_time'] = t['base_time'] + BDay(3)

            # 2️⃣ 再加 12 小时
            dt_plus_12h = t['base_time'] + pd.Timedelta(hours=12)

            # 3️⃣ 如果落在周末，顺延到下一个周一，保留时间
            if dt_plus_12h.weekday() >= 5:  # 5=周六，6=周日
                days_to_monday = 7 - dt_plus_12h.weekday()
                dt_plus_12h = dt_plus_12h + pd.Timedelta(days=days_to_monday)
            return dt_plus_12h
        else:
            return t['base_time'] + BDay(3)

    container_info["customs_del"] = container_info.apply(add_3_5_business_days,axis=1)

    #scf_trip_time=pd.read_excel('scf_trip_time.xlsx')
    scf_trip_time = pd.read_excel('scf_trip_time.xlsx')
    scf_transfer_time_dict=scf_trip_time.set_index('channel')['transfertime'].to_dict()
    container_info['scf_delivery_time'] =container_info['channel'].map(scf_transfer_time_dict).fillna(0)
    container_info = container_info.dropna(subset=['base_time','customs_del'])
    container_info['holiday']=container_info.apply(lambda t: "Y" if any((us_holiday_list >=t['base_time'].date()) & (us_holiday_list <=t['customs_del'].date())) else "N",axis=1)
    container_info["delivery_ddl"] = container_info.apply(
        lambda r: (
            r['customs_del'] if r['scf_type']=='0' else r['customs_del'] + pd.Timedelta(hours=r['scf_delivery_time'])
        ) if r['holiday'] == "N" else (
            r['customs_del'] + BDay(1) if r['scf_type']=='0' else r['customs_del'] + BDay(1) + pd.Timedelta(hours=r['scf_delivery_time'])
        ),
        axis=1
    )

    #SCF路程
    scf_route_time_dict=scf_trip_time.set_index('channel')['route_time'].to_dict()

    container_info['scf_route_time'] = container_info['channel'].map(scf_route_time_dict).fillna(0)

    container_info['outbound_ddl'] = container_info.apply(lambda r:r["delivery_ddl"] if r['scf_type']==0 else r["delivery_ddl"]-pd.Timedelta(hours=r['scf_route_time']) , axis=1)

    """
    根据运营时间（每天晚上9点关门，次日6点开门）计算仓库实际最晚出库时间
    """
    def adjust_time(t):
        if t.hour >= 18:
            return t.replace(hour=18, minute=0, second=0, microsecond=0)
        elif t.hour < 9:
            return (t - pd.Timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            return t
    container_info['outbound_ddl']=container_info['outbound_ddl'].apply(adjust_time)

    # 数据池
    df_pool=container_info.copy()
    # 模拟渠道-线路关系表
    df_route_map = pd.read_excel('channel_route.xlsx')

    # ==========================================
    # 2. 核心算法逻辑
    # ==========================================

    def smart_consolidation_with_state(df_data, df_map):
        # --- A. 预处理 ---
        # 关联映射表
        df_pool = pd.merge(df_data, df_map, on='channel', how='left')
        df_pool = df_pool.dropna(subset=['route', 'stops', 'state'])
        df_pool = df_pool.reset_index(drop=True)
        df_pool['is_shipped'] = False  # 标记位

        trucks_result = []
        truck_counter = 1

        # --- B. 主循环：直到发完所有货 ---
        while df_pool['is_shipped'].sum() < len(df_pool):

            # 1. 获取剩余数据
            remaining_mask = df_pool['is_shipped'] == False
            df_remaining = df_pool[remaining_mask].copy()

            if df_remaining.empty:
                break

            # 2. 寻找"锚点" (Anchor)
            # 规则：全局出库时间最早的那个 Stop 决定了这一车的"基调"
            earliest_idx = df_remaining['outbound_ddl'].idxmin()
            anchor_row = df_remaining.loc[earliest_idx]

            target_route = anchor_row['route']
            anchor_state = anchor_row['state']
            # 注意：这里我们只定线路和首发州，具体的 Stop 列表下面算

            # 3. 计算该线路下所有 Stop 的优先级 (核心逻辑)
            # 筛选出该线路下剩余的所有数据
            route_data = df_remaining[df_remaining['route'] == target_route]

            # 按 Stop 聚合，找出每个 Stop 最早的时间
            stop_stats = route_data.groupby(['stops', 'state'])['outbound_ddl'].min().reset_index()

            # --- 关键排序逻辑 ---
            # 优先级 1: State 是否与 Anchor_State 相同 (相同=0, 不同=1) -> 升序
            # 优先级 2: outbound_ddl (时间越早越前) -> 升序
            stop_stats['is_same_state'] = stop_stats['state'].apply(lambda x: 0 if x == anchor_state else 1)

            # 执行排序
            sorted_stops = stop_stats.sort_values(by=['is_same_state', 'outbound_ddl'])

            # 4. 开始装车
            current_truck = {
                'indices': [],
                'load': 0,
                'stops': set(),
                'states': set()  # 仅用于记录展示
            }

            # 遍历排序好的 Stop 列表 (先同州，后异州)
            for _, stop_row in sorted_stops.iterrows():
                current_stop = stop_row['stops']

                # [约束检查] Stop 数量限制
                # 如果是新 Stop，且已有 3 个了，则跳过该 Stop
                is_new_stop = current_stop not in current_truck['stops']
                if is_new_stop and len(current_truck['stops']) >= 3:
                    continue

                # 获取该 Stop 下的具体 Container，按时间排序
                candidates = df_remaining[
                    (df_remaining['route'] == target_route) &
                    (df_remaining['stops'] == current_stop)
                    ].sort_values(by='outbound_ddl')

                truck_full_flag = False

                for idx, row in candidates.iterrows():
                    box_count = row['boxCount']

                    # [约束检查] 体积限制
                    if current_truck['load'] + box_count <= 312:
                        current_truck['indices'].append(idx)
                        current_truck['load'] += box_count
                        current_truck['stops'].add(current_stop)
                        current_truck['states'].add(row['state'])
                    else:
                        # 车满了
                        truck_full_flag = True
                        break  # 停止装当前 Stop，同时也意味着整车结束

                if truck_full_flag:
                    break  # 停止遍历后续 Stop，发车

            # 5. 封车
            if current_truck['indices']:
                df_pool.loc[current_truck['indices'], 'is_shipped'] = True
                packed_data = df_pool.loc[current_truck['indices']]
                save_truck(trucks_result, packed_data, current_truck, truck_counter)
                truck_counter += 1
            else:
                break

        return pd.DataFrame(trucks_result)

    def save_truck(result_list, packed_df, truck_info, truck_id):
        stops_list = sorted([str(s) for s in list(truck_info['stops'])])
        states_list = sorted([str(s) for s in list(truck_info['states'])])

        result_list.append({
            '车次': f"{truck_id:03d}",
            '出库时间': packed_df['outbound_ddl'].min(),
            '线路': packed_df['route'].iloc[0],  # 取第一个即可
            '绑定的Container号码': ', '.join(packed_df['containerNo']),
            'Container数量': len(packed_df),
            '总大箱数': truck_info['load'],
            '渠道': ', '.join(sorted(packed_df['channel'].unique())),
            'Stop点个数': len(stops_list),
            'Stop点位': ', '.join(stops_list),
            '涉及州': ', '.join(states_list)  # 方便验证逻辑
        })

    # ==========================================
    # 3. 执行
    # ==========================================

    df_final = smart_consolidation_with_state(df_pool, df_route_map)

    # 格式化输出
    if not df_final.empty:
        df_final['出库时间'] = df_final['出库时间'].dt.strftime('%Y-%m-%d %H:%M')
    df_final.drop(columns='线路',axis=1,inplace=True)

    return df_final