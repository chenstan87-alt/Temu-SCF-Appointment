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
    container_info = container_[container_['containerNo'].str.startswith('99M', na=False)]
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
    us_holiday = pd.read_excel('/Users/stan/PycharmProjects/pythonProject/生产数据分析/时效数据/KPI prealert/USPS美国法定假日.xlsx')
    us_holiday_list=us_holiday['日期'].to_list()
    us_holiday_list = pd.to_datetime(us_holiday_list).date


    def add_3_5_business_days(t):
        # 先加 3 个工作日
        if t['channel_info']=="USPSGRR":
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
    scf_trip_time = pd.read_excel('/Users/stan/PycharmProjects/pythonProject/生产数据分析/时效数据/KPI prealert/scf_trip_time.xlsx')
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
    container_info['outbound_ddl']=container_info['outbound_ddl'].apply(adjust_time,axis=1)

    # 数据池 A
    df_pool_a=container_info.copy()
    # 模拟渠道-线路关系表
    df_route_map = pd.read_excel('channel_route.xlsx')

    # ==========================================
    # 2. 排序 (Step 2 )
    # ==========================================

    # 关联线路 (Left Join)
    df_pool_a = pd.merge(df_pool_a, df_route_map, on='channel', how='left')

    # 去除没有匹配到线路的数据 (可选，防止报错)
    df_pool_a = df_pool_a.dropna(subset=['route'])

    # 按照出库时间排序 (FIFO)
    df_pool_a = df_pool_a.sort_values(by='outbound_ddl')

    # ==========================================
    # 3. 核心拼车算法 (Step 3) - 限制 312 大箱
    # ==========================================

    MAX_BOXES_PER_TRUCK = 312
    trucks_result = []
    truck_counter = 1

    # 按线路分组处理
    for route, group_df in df_pool_a.groupby('route'):

        # 初始化当前车的缓存
        current_truck_containers = []
        current_truck_load = 0

        # 遍历该线路下的所有 Container
        for idx, row in group_df.iterrows():
            box_count = row['boxCount']

            # 判断：如果加上当前这个箱子，是否会超载？
            if current_truck_load + box_count > MAX_BOXES_PER_TRUCK:

                # --- A. 结算上一辆车 (如果缓存里有货) ---
                if current_truck_containers:
                    trucks_result.append({
                        '车次': f"{truck_counter:03d}",
                        '线路': route,
                        # 取本车中最早的出库时间
                        '出库时间': min(c['outbound_ddl'] for c in current_truck_containers),
                        # 拼接所有 Container No
                        '绑定的Container No': ', '.join([c['containerNo'] for c in current_truck_containers]),
                        # 拼接所有渠道 (去重)
                        '渠道': ', '.join(sorted(list(set([c['channel'] for c in current_truck_containers])))),
                        'nass code': ', '.join(sorted(list(set([c['m=nasscode'] for c in current_truck_containers])))),
                        '实际装载大箱数': current_truck_load
                    })
                    truck_counter += 1

                # --- B. 开启新车，并放入当前这个导致超载的箱子 ---
                current_truck_containers = [row]
                current_truck_load = box_count

            else:
                # --- C. 未超载，继续装入当前车 ---
                current_truck_containers.append(row)
                current_truck_load += box_count

        # --- D. 循环结束，处理最后一辆未满的车 ---
        if current_truck_containers:
            trucks_result.append({
                '车次': f"{truck_counter:03d}",
                '线路': route,
                '出库时间': min(c['outbound_ddl'] for c in current_truck_containers),
                '绑定的Container No': ', '.join([c['containerNo'] for c in current_truck_containers]),
                '渠道': ', '.join(sorted(list(set([c['channel'] for c in current_truck_containers])))),
                'nass code': ', '.join(sorted(list(set([c['m=nasscode'] for c in current_truck_containers])))),
                '实际装载大箱数': current_truck_load
            })
            truck_counter += 1

    # ==========================================
    # 4. 输出结果 (Step 4)
    # ==========================================

    df_final = pd.DataFrame(trucks_result)

    # 格式化时间列，看起来更整洁
    df_final['出库时间'] = df_final['出库时间'].dt.strftime('%Y-%m-%d %H:%M')

    return df_final