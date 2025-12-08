import os
import json
import sys

#this will import env variables
from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

#to ensure trusted certificates, all libraries that we use, mongodb understands that secure http connection is established
import certifi
ca=certifi.where() #retrieves bundle of al certified authorities, stored in ca to establish secure ssl tsh connection with server(my comp) mongobd understand this

import pandas as pd
import numpy as np
import pymongo
from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging

class TelecomChurnDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise TelecomChurnDataExtract(e,sys)
    
    #conver the csv into json
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            # remove index of csv
            data.reset_index(drop=True, inplace=True)
            # refer notes
            records= list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise TelecomChurnDataExtract(e,sys)
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.records=records
            self.database=database
            self.collection=collection

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL,tlsCAFile=ca)

            self.database=self.mongo_client[self.database]
            self.collection=self.database[collection]

            self.collection.insert_many(self.records)

            return (len(self.records))

        except Exception as e:
            raise TelecomChurnDataExtract(e,sys)

if __name__=="__main__":
    FILE_PATH="Churn Data\Churn_prediction_cleaned.csv"
    DATABASE="Samiulla_Churn"
    COLLECTION="TelecomChurn"
    telecomobj=TelecomChurnDataExtract()
    records=telecomobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=telecomobj.insert_data_mongodb(records,DATABASE,COLLECTION)
    print(no_of_records)






