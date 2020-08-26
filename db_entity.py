import urllib.parse
from pymongo import MongoClient
import numpy as np


class DBEntity():
    def __init__(self):
        username = urllib.parse.quote_plus('mongo')
        password = urllib.parse.quote_plus('mongo') 
        self.conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
        self.db = self.conn["PoseMachine"]

    def convert_np_array_to_list(self, np_array):
        list_arr = np.array(np_array).tolist()
        return list_arr

    def insert_average_angles(self, angles: list, class_name: str):
        collection = self.db["average_angles"]
        angles = self.convert_np_array_to_list(angles)
        try:
            average_angles_dict = {
                'angles': angles,
                'class_name': class_name,
            }
            collection.insert_one(average_angles_dict) 
        except Exception as e:
            print(e)
        

'''
insert_threshold_to_db

Insert exercise movements threshold into Mongo DB
@params {float} x_threshold
@params {float} y_threshold
@params {float} class of exercise
'''
def insert_threshold_to_db(x_threshold, y_threshold, class_name, keypoint_dict):
    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo')
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[class_name + "-threshold"]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            # Make new object
            pose = {
                "keypoint_name": keypoint_dict["name"],
                "keypoint_num": keypoint_dict["value"],
                "x_threshold": x_threshold,
                "y_threshold": y_threshold
            }

            # Insert into database collection
            rec_id1 = collection.insert_one(pose) 
            print("Data inserted with record ids",rec_id1) 
        except Exception as e:
            print("Failed to insert data to database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 

'''
insert_array_to_db

Insert array of poses into Mongo DB
@params {list} list_of_pose
@params {float} class of exercise
'''
def insert_array_to_db(list_of_pose, class_type, class_name):
    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[class_name]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            # Make new object
            pose = {
                "list_of_pose": list_of_pose,
                "exercise_type": class_type
            }

            # Insert into database collection
            rec_id1 = collection.insert_one(pose) 
            print("Data inserted with record ids",rec_id1) 
        except Exception as e:
            print("Failed to insert data to database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 

'''
insert_np_array_to_db

Insert numpy array of poses into Mongo DB
@params {list} list_of_pose
@params {float} class of exercise
'''
def insert_np_array_to_db(list_of_pose, class_type, class_name):
    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[class_name]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # Initialize temp lists
        list_of_pose_arr = []
        # try catch for MongoDB insert
        try:
            # list_of_pose consists of poses which is type of np array
            # Loop to get np array inside a normal array 
            # (mongodb does not accept np array)
            for pose in list_of_pose:
                list_of_pose_arr.append(pose.tolist())
                
            # Make new object
            pose = {
                "list_of_pose": list_of_pose_arr,
                "exercise_type": class_type
            }

            # Insert into database collection
            rec_id1 = collection.insert_one(pose) 
            print("Data inserted with record ids",rec_id1) 
        except Exception as e:
            print("Failed to insert data to database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 

'''
get_dataset

# Get data from MongoDB
'''
# Get from Mongo DB
def get_dataset(collection_name):
    # Initialize temp lists
    list_of_poses = []
    list_of_labels = []

    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[collection_name]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            for x in collection.find():
                list_of_poses.append(x["list_of_pose"])
                list_of_labels.append(x["exercise_type"])
        except Exception as e:
            print("Failed to get data from database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 
    
    return list_of_poses, list_of_labels

'''
get_dataset_with_limit

# Get data from MongoDB with limit
'''
# Get from Mongo DB
def get_dataset_with_limit(collection_name, limit):
    # Initialize temp lists
    list_of_poses = []
    list_of_labels = []

    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[collection_name]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            for x in collection.find():
                list_of_poses.append(x["list_of_pose"])
                list_of_labels.append(x["exercise_type"])
                if len(list_of_poses) == limit:
                    break
        except Exception as e:
            print("Failed to get data from database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 
    
    return list_of_poses, list_of_labels

'''
get_count

# Get count from MongoDB collection
'''
# Get from Mongo DB
def get_count(collection_name):
    # Initialize temp lists
    list_of_poses = []
    list_of_labels = []

    # Initialize count
    count = 0

    # try catch for MongoDB connection
    try: 
        #connect to mongodb instance
        username = urllib.parse.quote_plus('mongo') 
        password = urllib.parse.quote_plus('mongo') 
        conn = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))

        # connect to mongodb database and collection
        db = conn["PoseMachine"]
        collection = db[collection_name]
        
        # If successful print
        print("\nConnected successfully!!!\n") 

        # try catch for MongoDB insert
        try:
            count = collection.count()
        except Exception as e:
            print("Failed to get data from database, errors: ", e) 
                        
    except Exception as e:   
        print("Could not connect to MongoDB " , e) 
    
    return count