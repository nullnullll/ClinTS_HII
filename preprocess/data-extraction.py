import pickle
import psycopg2 as py


# Replace this with your mimic iii database details
conn = py.connect(
    "dbname = 'mimic3' user = 'covpreduser' host = 'localhost' password = '2021' options='-c search_path=mimiciii' ")

cur = conn.cursor()
cur.execute("""select hadm_id from admissions """)
list_adm_id = cur.fetchall()

cur.execute("select hadm_id, admission_type, trunc(extract(epoch from " +
            "dischtime- admittime)/3600), hospital_expire_flag from admissions ")
length_of_stay = cur.fetchall()
pickle.dump(length_of_stay, open('adm_type_los_mortality.p', 'wb'))


data = []

for id in range(len(list_adm_id)):
    print("len:"+str(len(list_adm_id)),id, list_adm_id[id][0])
    patient = []

    # print("Sp02")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(646) +
                "or itemid =" + str(220277) + ")order by charttime")
    patient.append(cur.fetchall())

    # Heart Rate
    # print("HR")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(211) +
                "or itemid =" + str(220045) + ")order by charttime")
    patient.append(cur.fetchall())

    # Respiratory Rate
    # print("RR")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(618) +
                "or itemid =" + str(615) + "or itemid =" + str(220210) +
                "or itemid =" + str(224690) + ")order by charttime")
    patient.append(cur.fetchall())

    # Systolic Blood Pressure
    # print("SBP")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(51) +
                "or itemid =" + str(442) + "or itemid =" + str(455) +
                "or itemid =" + str(6701) + "or itemid =" + str(220179) +
                "or itemid =" + str(220050) + ")order by charttime")
    patient.append(cur.fetchall())

    # Diastolic Blood Pressure
    # print("DBP")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(8368) +
                "or itemid =" + str(8440) + "or itemid =" + str(8441) +
                "or itemid =" + str(8555) + "or itemid =" + str(220180) +
                "or itemid =" + str(220051) + ")order by charttime")
    patient.append(cur.fetchall())

    # End-tidal carbon dioxide
    # print("EtC02")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(1817) +
                "or itemid =" + str(228640) + ")order by charttime")
    patient.append(cur.fetchall())

    # Temperature
    # print("Temp")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(678) +
                "or itemid =" + str(223761) + ")order by charttime")
    patient.append(cur.fetchall())
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(676) +
                "or itemid =" + str(223762) + ")order by charttime")
    patient.append(cur.fetchall())

   # Total Glasgow coma score
    # print("TGCS")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(198) +
                "or itemid =" + str(226755) + "or itemid =" + str(227013)
                + ")order by charttime")
    patient.append(cur.fetchall())

    # Peripheral capillary refill rate
    # print("CRR")
    cur.execute("select charttime, value from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and itemid =" + str(3348) +
                "order by charttime")
    patient.append(cur.fetchall())
    cur.execute("select charttime, value from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(115) +
                "or itemid = 223951) order by charttime")
    patient.append(cur.fetchall())
    cur.execute("select charttime, value from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(8377) +
                "or itemid = 224308) order by charttime")
    patient.append(cur.fetchall())

    # Urine output 43647, 43053, 43171, 43173, 43333, 43347, 43348, 43355, 43365, 43373, 43374, 43379
    # print("UO")
    cur.execute("select charttime, VALUE from outputevents where hadm_id ="
                + str(list_adm_id[id][0]) + " and ( itemid = 40405 or itemid =" +
                " 40428 or itemid = 41857 or itemid = 42001 or itemid = 42362 or itemid =" +
                " 42676 or itemid = 43171 or itemid = 43173 or itemid = 42042 or itemid =" +
                " 42068 or itemid = 42111 or itemid = 42119 or itemid = 40715 or itemid =" +
                " 40056 or itemid = 40061 or itemid = 40085 or itemid = 40094 or itemid =" +
                " 40096 or itemid = 43897 or itemid = 43931 or itemid = 43966 or itemid =" +
                " 44080 or itemid = 44103 or itemid = 44132 or itemid = 44237 or itemid =" +
                " 43348 or itemid =" +
                " 43355 or itemid = 43365 or itemid = 43372 or itemid = 43373 or itemid =" +
                " 43374 or itemid = 43379 or itemid = 43380 or itemid = 43431 or itemid =" +
                " 43462 or itemid = 43522 or itemid = 44706 or itemid = 44911 or itemid =" +
                " 44925 or itemid = 42810 or itemid = 42859 or itemid = 43093 or itemid =" +
                " 44325 or itemid = 44506 or itemid = 43856 or itemid = 45304 or itemid =" +
                " 46532 or itemid = 46578 or itemid = 46658 or itemid = 46748 or itemid =" +
                " 40651 or itemid = 40055 or itemid = 40057 or itemid = 40065 or itemid =" +
                " 40069 or itemid = 44752 or itemid = 44824 or itemid = 44837 or itemid =" +
                " 43576 or itemid = 43589 or itemid = 43633 or itemid = 43811 or itemid =" +
                " 43812 or itemid = 46177 or itemid = 46727 or itemid = 46804 or itemid =" +
                " 43987 or itemid = 44051 or itemid = 44253 or itemid = 44278 or itemid =" +
                " 46180 or itemid = 45804 or itemid = 45841 or itemid = 45927 or itemid =" +
                " 42592 or itemid = 42666 or itemid = 42765 or itemid = 42892 or itemid =" +
                " 43053 or itemid = 43057 or itemid = 42130 or itemid = 41922 or itemid =" +
                " 40473 or itemid = 43333 or itemid = 43347 or itemid = 44684 or itemid =" +
                " 44834 or itemid = 43638 or itemid = 43654 or itemid = 43519 or itemid =" +
                " 43537 or itemid = 42366 or itemid = 45991 or itemid = 43583 or itemid =" +
                " 43647) order by charttime ")
    patient.append(cur.fetchall())

    # Fraction inspired oxygen
    # print("Fi02")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(2981) +
                "or itemid =" + str(3420) + "or itemid =" + str(3422) +
                "or itemid =" + str(223835) + ")order by charttime")
    patient.append(cur.fetchall())

    # Glucose 807,811,1529,3745,3744,225664,220621,226537
    # print("Glucose")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(807) +
                "or itemid =" + str(811) + "or itemid =" + str(1529) +
                "or itemid =" + str(3745) + "or itemid =" + str(3744) +
                "or itemid =" + str(225664) + "or itemid =" + str(220621) +
                "or itemid =" + str(226537) + ")order by charttime")
    patient.append(cur.fetchall())

    # pH 780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243,
    # print("pH")
    cur.execute("select charttime, valuenum from chartevents where hadm_id ="
                + str(list_adm_id[id][0]) + "and (itemid =" + str(780) +
                "or itemid =" + str(860) + "or itemid =" + str(1126) +
                "or itemid =" + str(1673) + "or itemid =" + str(3839) +
                "or itemid =" + str(4202) + "or itemid =" + str(4753) +
                "or itemid =" + str(6003) + "and itemid =" + str(220274) +
                "or itemid =" + str(220734) + "or itemid =" + str(223830) +
                "or itemid =" + str(228243) + ") order by charttime")
    patient.append(cur.fetchall())

    # mechanical ventilation
    cur.execute(
        "select   charttime,max(case when itemid is null or value is null then 0 when itemid = 720 and value != 'Other/Remarks' THEN 1 when itemid = 223848 and value != 'Other' THEN 1 when itemid = 223849 then 1 when itemid = 467 and value = 'Ventilator' THEN 1 when itemid in(445, 448, 449, 450, 1340, 1486, 1600, 224687 ,639, 654, 681, 682, 683, 684, 224685, 224684, 224686, 218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706 , 60, 437, 505, 506, 686, 220339, 224700 , 3459 , 501, 502, 503, 224702 , 223, 667, 668, 669, 670, 671, 672 , 224701) THEN 1 else 0 end ) as MechVent " +
        " from chartevents where hadm_id =" + str(list_adm_id[id][0]) +
        "and value is not null " +
        "and (error != 1 or error IS NULL) " +
        "and itemid in" +
        "(720, 223849, 223848 , 445, 448, 449, 450, 1340, 1486, 1600, 224687 , 639, 654, 681, 682, 683, 684,224685,224684,224686 , 218,436,535,444,224697,224695,224696,224746,224747, 221,1,1211,1655,2000,226873,224738,224419,224750,227187 , 543 , 5865,5866,224707,224709,224705,224706 , 60,437,505,506,686,220339,224700 , 3459, 501,502,503,224702, 223,667,668,669,670,671,672 , 224701, 640 , 468 , 469 , 470, 471 , 227287 , 226732, 223834 , 467) " +
        "group by  charttime")
    patient.append(cur.fetchall())

    # vasopressor
    cur.execute("with io_cv as( select icustay_id, charttime, itemid, stopped, "
                "case when itemid in (42273, 42802) then amount else rate end as rate, "
                "case when itemid in (42273, 42802) then rate else amount end as amount " +
                "FROM inputevents_cv " +
                "where hadm_id = " + str(list_adm_id[id][
                                             0]) + " AND itemid in(30047,30120,30044,30119,30309,30127, 30128,30051,30043,30307,30042,30306,30125, 42273, 42802)) " +
                "select charttime, 1 as vaso from io_cv  group by  charttime"
                )
    patient.append(cur.fetchall())

    # adenosine
    cur.execute("select charttime, max(case when itemid = 4649 then 1 else 0 end) as vaso " +
                "FROM chartevents " +
                "where hadm_id = " + str(list_adm_id[id][0]) + " AND " +
                "itemid = 4649 AND (error IS NULL OR error = 0) " +
                "group by  charttime")
    patient.append(cur.fetchall())

    # dobutamine
    cur.execute("select charttime , max(case when itemid in (30042,30306) then 1 else 0 end) as vaso " +
                " FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in (30042,30306) " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # dopamine
    cur.execute("select charttime , max(case when itemid in (30043,30307) then 1 else 0 end) as vaso " +
                " FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in (30043,30307) " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # epinephrine
    cur.execute("select charttime , max(case when itemid in (30044,30119,30309) then 1 else 0 end) as vaso " +
                " FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in (30044,30119,30309) " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # isuprel
    cur.execute("select charttime , max(case when itemid = 30046 then 1 else 0 end) as vaso " +
                "FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid = 30046 " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # milrinone
    cur.execute("select charttime , max(case when itemid = 30125 then 1 else 0 end) as vaso " +
                "FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid = 30125 " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # norepinephrine
    cur.execute("select charttime , max(case when itemid in (30047,30120) then 1 else 0 end) as vaso " +
                "FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in (30047,30120) " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # phenylephrine
    cur.execute("select charttime , max(case when itemid in (30127,30128) then 1 else 0 end) as vaso " +
                "FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in (30127,30128) " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # vasopressin
    cur.execute("select charttime , max(case when itemid = 30051 then 1 else 0 end) as vaso " +
                "FROM inputevents_cv " +
                "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid = 30051 " +
                "group by charttime"
                )
    patient.append(cur.fetchall())

    # colloid_bolus
    cur.execute(
        "with t1 as (select  starttime as charttime, round(case when amountuom = 'L' then amount * 1000.0 " +
        "when amountuom = 'ml' then amount else null end) as amount from inputevents_mv " +
        "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in(220864, 220862, 225174, 225795, 225796) " +
        "and statusdescription != 'Rewritten' " +
        "and((rateuom = 'mL/hour' and rate > 100) OR (rateuom = 'mL/min' and rate > (100/60.0)) OR (rateuom = 'mL/kg/hour' and (rate*patientweight) > 100)) ) " +
        ", t2 as (select  charttime, round(amount) as amount " +
        "from inputevents_cv " +
        "where hadm_id =" + str(list_adm_id[id][
                                    0]) + " AND itemid in (30008 ,30009 ,42832 ,40548 ,45403 ,44203,30181,46564 ,43237,43353,30012  ,46313 ,30011,30016,42975,42944,46336,46729,40033,45410,42731 ) " +
        "and amount > 100 and amount < 2000) " +
        ", t3 as (select charttime, round(valuenum) as amount " +
        "from chartevents " +
        "where hadm_id =" + str(list_adm_id[id][0]) + " AND itemid in ( 2510 , 3087 , 6937 , 3087 , 3088 ) " +
        "and valuenum is not null and valuenum > 100 and valuenum < 2000) " +
        "select  charttime, sum(amount) as colloid_bolus from t1 " +
        "where  amount > 100 group by  t1.charttime " +
        "UNION ALL " +
        "select  charttime, sum(amount) as colloid_bolus from t2 " +
        "group by  t2.charttime " +
        "UNION ALL " +
        "select  charttime, sum(amount) as colloid_bolus from t3 " +
        "group by  t3.charttime order by  charttime "
    )
    patient.append(cur.fetchall())

    # crystalloid_bolus
    cur.execute(
        "with t1 as (select  starttime as charttime, round(case when amountuom = 'L' then amount * 1000.0 " +
        "when amountuom = 'ml' then amount else null end) as amount from inputevents_mv " +
        "where hadm_id =" + str(list_adm_id[id][
                                    0]) + " AND itemid in(225158, 225828,225944, 225797, 225159, 225823, 225825, 225827, 225941,226089) " +
        "and statusdescription != 'Rewritten' " +
        "and((rate is not null and rateuom = 'mL/hour' and rate > 248)OR (rate is not null and rateuom = 'mL/min' and rate > (248/60.0))OR (rate is null and amountuom = 'L' and amount > 0.248)OR (rate is null and amountuom = 'ml' and amount > 248)) ) " +
        ", t2 as (select  charttime, round(amount) as amount " +
        "from inputevents_cv " +
        "where hadm_id =" + str(list_adm_id[id][
                                    0]) + " AND itemid in (30015 , 30018 , 30020  , 30021  , 30058  , 30060  , 30061  , 30063 , 30065  , 30143 , 30159  , 30160  , 30169  , 30190  , 40850  , 41491 , 42639  , 42187  , 43819  , 41430  , 40712  , 44160  , 42383  , 42297  , 42453  , 40872  , 41915  , 41490  , 46501  , 45045  , 41984  , 41371  , 41582  , 41322  , 40778  , 41896  , 41428  , 43936  , 44200  , 41619  , 40424  , 41457  , 41581  , 42844  , 42429  , 41356  , 40532  , 42548  , 44184  , 44521  , 44741  , 44126  , 44110 , 44633  , 44983   , 44815  , 43986  , 45079   , 46781 , 45155  , 43909  , 41467  , 44367 , 41743  , 40423  , 44263  , 42749  , 45480  , 44491  , 41695  , 46169  , 41580  , 41392  , 45989  , 45137  , 45154  , 44053  , 41416  , 44761   , 41237 , 44426  , 43975  , 44894  , 41380  , 42671  ) " +
        "and amount > 248 and amount <= 2000 and amountuom = 'ml' )" +

        "select  charttime, sum(amount) as crystalloid_bolus from t1 " +
        "where  amount > 248 group by  t1.charttime " +
        "UNION " +
        "select  charttime, sum(amount) as crystalloid_bolus from t2 " +
        "group by  t2.charttime order by charttime"
    )
    patient.append(cur.fetchall())

    data.append(patient)


pickle.dump(data, open('./data/patient_records.p', 'wb'))
