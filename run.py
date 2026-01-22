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
    # 2. 核心算法函数
    # ==========================================

    def dynamic_consolidation(df_data, df_map):
        # --- 预处理 ---
        # 关联线路信息
        df_pool = pd.merge(df_data, df_map, on='channel', how='left')
        df_pool = df_pool.dropna(subset=['route', 'stops'])  # 剔除无效数据

        # 确保每个 container 都有唯一索引，方便后续剔除
        df_pool = df_pool.reset_index(drop=True)
        df_pool['is_shipped'] = False  # 标记是否已发货

        trucks_result = []
        truck_counter = 1

        # --- 主循环：只要还有未发货的 Container，就继续 ---
        while df_pool['is_shipped'].sum() < len(df_pool):

            # 1. 获取当前剩余的数据 (Pool View)
            remaining_mask = df_pool['is_shipped'] == False
            df_remaining = df_pool[remaining_mask].copy()

            if df_remaining.empty:
                break

            # 2. 计算动态优先级 (Dynamic Priority)
            # 统计每个 Stop 当前最早的 DDL
            stop_stats = df_remaining.groupby(['route', 'stops'])['outbound_ddl'].min().reset_index()
            stop_stats = stop_stats.rename(columns={'outbound_ddl': 'priority_time'})
            # 按时间排序，这一步决定了当前这辆车"最想"去哪个 Stop
            stop_priority_list = stop_stats.sort_values(by='priority_time')

            # 3. 初始化新车
            current_truck = {
                'indices': [],  # 记录装了哪些行的索引
                'load': 0,
                'stops': set(),
                'route': None  # 这一车绑定的线路
            }

            # 4. 尝试按优先级装货
            # 遍历排好序的 Stop 列表
            for _, priority_row in stop_priority_list.iterrows():
                target_route = priority_row['route']
                target_stop = priority_row['stops']

                # [规则 A] 线路一致性检查
                # 如果车还是空的，确定线路；如果车不空，必须和车的线路一致
                if current_truck['route'] is None:
                    current_truck['route'] = target_route
                elif current_truck['route'] != target_route:
                    continue  # 线路不同，不能拼，看下一个 Stop

                # [规则 B] Stop 数量检查
                # 如果是新 Stop，且车里已经有3个 Stop 了，跳过
                is_new_stop = target_stop not in current_truck['stops']
                if is_new_stop and len(current_truck['stops']) >= 3:
                    continue

                    # --- 开始装载该 Stop 的货 ---
                # 找出该 Stop 下剩余的所有 Container，并按时间排序 (内部 FIFO)
                candidates = df_remaining[
                    (df_remaining['route'] == target_route) &
                    (df_remaining['stops'] == target_stop)
                    ].sort_values(by='outbound_ddl')

                truck_is_full = False

                for idx, row in candidates.iterrows():
                    box_count = row['boxCount']

                    # [规则 C] 体积检查 (Max 312)
                    if current_truck['load'] + box_count <= 312:
                        # 装入
                        current_truck['indices'].append(idx)
                        current_truck['load'] += box_count
                        current_truck['stops'].add(target_stop)
                    else:
                        # 装不下了 -> 车满了
                        truck_is_full = True
                        break  # 停止装载当前 Stop，同时也意味着整车结束

                if truck_is_full:
                    break  # 停止遍历后续 Stop，发车

            # 5. 封车与标记
            if current_truck['indices']:
                # 在主表中标记这些 Container 为已发货
                df_pool.loc[current_truck['indices'], 'is_shipped'] = True

                # 收集结果数据
                packed_data = df_pool.loc[current_truck['indices']]
                save_truck(trucks_result, packed_data, current_truck, truck_counter)
                truck_counter += 1
            else:
                # 防死循环：如果找不到任何可装的货(逻辑上不太可能，除非所有货都超大无法装入空车)，强制退出
                break

        return pd.DataFrame(trucks_result)

    def save_truck(result_list, packed_df, truck_info, truck_id):
        stops_list = sorted([str(s) for s in list(truck_info['stops'])])
        result_list.append({
            '车次': f"{truck_id:03d}",
            '出库时间': packed_df['outbound_ddl'].min(),  # 这一车最早的时间
            'Stop数量': len(stops_list),
            'Stop点位': ', '.join(stops_list),
            '总大箱数': truck_info['load'],
            'Container数量': len(packed_df),
            'Container号码': ', '.join(packed_df['containerNo']),
            '渠道': ', '.join(sorted(packed_df['channel'].unique()))
        })

    # ==========================================
    # 3. 执行与输出
    # ==========================================

    df_final = dynamic_consolidation(df_pool, df_route_map)

    return df_final