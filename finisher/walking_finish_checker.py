import pymysql
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine

def transform_vector(user_vector, db_vector):
    N, _ = user_vector.shape
    M, _ = db_vector.shape
    K = min(N, M)
    Q = max(N, M)
    if K==N :
        vector = db_vector
    else : 
        vector = user_vector
    selected_indices = [0]  
    for i in range(1, K - 1):
        idx = int((i / (K - 1)) * (Q - 1)) + 1
        selected_indices.append(idx)
    
    selected_indices.append(K - 1)  
    
    
    transformed_vector = vector[selected_indices]
    
    if N == M :
        return user_vector, db_vector
    if N > M:
        return transformed_vector, db_vector
    else:
        return transformed_vector, user_vector
    
class  finish_Checker():
    def __init__(self, courseid):
        self.courseid = courseid

    def calculate_Similarity(self, input_coord) :
        # 두 산책로의 유사도가 높으면 false/true를 반환 하는 함수
        token_true = {
            "success" : True
        }
        token_false = {
            "success" : False
        }

        # 정규화 (X_mean, Y_mean : [ 37.554812 126.988204] X_std, Y_std :  [0.0031548  0.00720859])
        # jupyter notebook에서 standard scaler를 이용해서 얻을 결과를 값만 사용
        X_mean, Y_mean = 37.554812, 126.988204
        X_std, Y_std = np.sqrt(0.0031548), np.sqrt(0.00720859)

        # user가 등록하고자 하는 gps의 json을 좌표형식으로 쪼갠다.
        temp = np.array(input_coord.split())
        user_input = np.array([])
        for i in temp:
            user_input = np.append(user_input, float(i))
        user_input = user_input.reshape(-1, 2)
        user_XY = tuple(np.mean(user_input, axis=0))

        # 위에서 얻은 좌표형식의 gps값을 정규화
        user_frame = pd.DataFrame(user_input)
        user_frame.iloc[:, 0] = (user_frame.iloc[:, 0] - X_mean) / X_std
        user_frame.iloc[:, 1] = (user_frame.iloc[:, 1] - Y_mean) / Y_std
        user_std = user_frame.values

        # DB연결
        conn = pymysql.connect(host="172.20.0.5", user="root", password="1234", db="naemansan")

        cursor = conn.cursor()

        # DB에서 유저가 이용한 산책로를 모두 갖고온다.
        query = """
        SELECT ST_AsText(locations) 
        FROM enrollment_courses
        WHERE id = %d""" %(self.courseid)
        cursor.execute(query)
        results = cursor.fetchall()
        
        locations_list = []  # multipoint형 리스트
        coordinates_list = []  # multipoint를 float으로 변환한 리스트

        # multipoint 저장
        for row in results:
            locations_list.append(row[0])

        # float 형태로 저장
        for row in locations_list:
            row = row.replace("MULTIPOINT", "")
            row = row.replace("(", "")
            row = row.replace(")", "")
            row = row.replace(",", " ")
            temp = row.split()
            float_coord = []
            for i in temp[::-1]:  
                float_coord.append(float(i))
            coordinates_list.append(float_coord)

        # 좌표를 데이터프레임으로 변환
        walking_Path = pd.DataFrame(coordinates_list)

        # 산책로 마다 데이터프레임으로 변환
        X, Y = walking_Path.shape
        for i in range(X):
            walking = walking_Path.iloc[i]
            walking = walking.dropna()
            walking_list = walking.values
            walking_list = walking_list.reshape(-1, 2)
            temp_frame = pd.DataFrame(walking_list)
            temp_frame = temp_frame.dropna(axis=1)
            original_frame = temp_frame.copy()

            # 정규화
            temp_frame.iloc[:, 0] = (temp_frame.iloc[:, 0] - X_mean) / X_std 
            temp_frame.iloc[:, 1] = (temp_frame.iloc[:, 1] - Y_mean) / Y_std
            DB_std = temp_frame.values
            DB_starting_std = temp_frame.iloc[0].values
            

            #user_final = np.delete(user_result, 0, axis=0)
            #DB_final = np.delete(DB_result, 0, axis=0)

            # 유사도 벡터와 점수
            DB_XY = np.mean(original_frame, axis=0)
            DB_coord = (DB_XY[:][0], DB_XY[:][1])
            distance = haversine(user_XY, DB_coord)
            left_vector, right_vector = transform_vector(user_std, DB_std)


            disp_left = (np.zeros((1,2)) - (left_vector[0,0],left_vector[0,1]))
            disp_right = (np.zeros((1,2)) - (right_vector[0,0],right_vector[0,1]))
            left_vector[:,0] += disp_left[0][0]
            left_vector[:,1] += disp_left[0][1]
            right_vector[:,0] += disp_right[0][0]
            right_vector[:,1] += disp_right[0][1]

            left_vector = left_vector[1:]
            right_vector = right_vector[1:]

            similarity_vector = cosine_similarity(left_vector, right_vector)
            similarity_size = similarity_vector.shape[0]

            similarity_score = np.trace((similarity_vector)) / similarity_size
            #print(similarity_vector)
            threshold = 0.90

            # 유사도가 높으면 반복문 멈추고 등록 불가
            # 산책로의 좌표가 길면 유사도 벡터가 점점 희소해지는 문제를 벡터 열의 max값만 사용
            # 벡터 열의 max값을 이용한다는 것의 의미는 A 산책로의 한 좌표와 제일 가까운 B 산책로의 한 좌표와의 유사도를 의미
            if (similarity_score > threshold or similarity_score < -threshold) and distance < 0.5 :
                return token_true

        return token_false