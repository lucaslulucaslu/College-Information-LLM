import os

import pandas as pd
import pymysql

from utilities.schema import RankingType


class CollegeRanking:
    def __init__(self):
        pass

    def get_ranking_types():
        # Connect to the database
        connection = pymysql.connect(
            db=os.environ["db_name"],
            user=os.environ["db_user"],
            passwd=os.environ["db_pass"],
            host=os.environ["db_host"],
            port=3306,
            cursorclass=pymysql.cursors.DictCursor,
        )

        # Query the database
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT `school`,`level`,max(`year`) as year FROM fp_ranking.`major_rank_table` WHERE `program` IS NULL\
                    GROUP BY `school`,`level`"
            )
            ranking_types = cursor.fetchall()
            ranking_types_df = pd.DataFrame(ranking_types)
        # Close the connection
        connection.close()

        return ranking_types_df

    def get_usnews_ranking():
        # Connect to the database
        connection = pymysql.connect(
            db=os.environ["db_name"],
            user=os.environ["db_user"],
            passwd=os.environ["db_pass"],
            host=os.environ["db_host"],
            port=3306,
            cursorclass=pymysql.cursors.DictCursor,
        )

        # Query the database
        with connection.cursor() as cursor:
            cursor.execute("""SELECT ranking FROM fp_IPEDS.latest_information""")
            row = cursor.fetchone()
            ranking_year = row["ranking"]

            cursor.execute(
                """SELECT t1.rank, t2.cname, t2.name as ename 
                   FROM fp_ranking.us_rankings t1
                   JOIN fp_ranking.colleges t2 ON t2.postid = t1.postid
                   WHERE t1.year = (SELECT MAX(year) FROM fp_ranking.us_rankings WHERE type = 1) 
                   AND t1.type = 1
                   ORDER BY t1.rank ASC 
                   LIMIT 50"""
            )
            usnews_ranking = cursor.fetchall()
            usnews_ranking_df = pd.DataFrame(usnews_ranking)

        # Close the connection
        connection.close()
        return (ranking_year, usnews_ranking_df)

    def get_major_ranking(ranking_type: RankingType):
        # Connect to the database
        connection = pymysql.connect(
            db=os.environ["db_name"],
            user=os.environ["db_user"],
            passwd=os.environ["db_pass"],
            host=os.environ["db_host"],
            port=3306,
            cursorclass=pymysql.cursors.DictCursor,
        )

        # Query the database
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM fp_ranking.`major_rank_table` WHERE `school` = '{ranking_type.school}' \
                    AND `program` IS NULL AND `level` = '{ranking_type.level}' ORDER BY `year` DESC LIMIT 1"
            )
            row = cursor.fetchone()
            rank_table_id = row["ID"]
            cursor.execute(
                f"""SELECT t1.rank,t1.post_ID,IF(t2.cname IS NOT NULL,t2.cname,t1.backupcname) as cname,\
                    IF(t2.name IS NOT NULL,t2.name,t1.backupname) as ename FROM fp_ranking.`major_rank_relation` t1
                    LEFT JOIN fp_ranking.colleges t2 ON t1.post_ID=t2.postid WHERE t1.`rank_table_ID` = {rank_table_id} \
                        ORDER BY t1.`rank` ASC LIMIT 50"""
            )
            ranking_data = cursor.fetchall()
            ranking_df = pd.DataFrame(ranking_data)
        # Close the connection
        connection.close()

        return ranking_df
