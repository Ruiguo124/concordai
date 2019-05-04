import psycopg2
import pickle
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def sql_conn():
    connection = psycopg2.connect(user="kyle",
                                  password="q",
                                  host="192.168.43.190",
                                  port="5432",
                                  database="db1")
    return connection


def sql_query(query):
    connection = sql_conn()

    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print(connection.get_dsn_parameters(), "\n")
    # Print PostgreSQL version

    cursor.execute(query)
    records = cursor.fetchall()

    cursor.close()
    connection.close()
    return records


if __name__ == "__main__":
    if not os.path.isfile("labels.p") or not os.path.isfile("X_dataset.p"):
        labels = set()
        mobile_records = sql_query(
            "SELECT trips.id_trip, mode, points.speed, points.timestamp, points.latitude, points.longitude, points.id_trip, points.altitude,trips.starttime, trips.endtime, points.v_accuracy  FROM trips INNER JOIN points ON points.id_trip = trips.id_trip WHERE mode ~ '^[a-zA-Z / ]+$' AND mode NOT LIKE 'ND' ORDER BY trips.id_trip ASC, points.timestamp ASC")

        length = len(mobile_records)
        finalArray = []
        tempArray = []
        modeArray = []

        numberOfPoints = 0.0
        totalAcceleration = 0.0
        totalDistance = 0.0
        totalSpeed = 0.0
        numberOfTrips = 0
        totaltime = 0

        for i in range(0, length - 1):

            # works: Distance
            distance = (mobile_records[i][4] ** 2 + mobile_records[i][5] ** 2) ** (0.5)
            # print("distance = ", distance, "\n")

            totalDistance += distance * mobile_records[i][10]

            # Acceleration

            if (mobile_records[i][0] == mobile_records[i + 1][0]):
                if (((mobile_records[i + 1][3] - mobile_records[i][3]).total_seconds()) == 0):
                    acceleration = 0
                    continue
                acceleration = ((mobile_records[i + 1][2] / 3.6) - (mobile_records[i][2]) / 3.6) / (
                        mobile_records[i + 1][3] - mobile_records[i][3]).total_seconds()
                totalAcceleration += acceleration
                numberOfPoints = numberOfPoints + 1
                # print("acceleration is: ", acceleration)
            else:
                if numberOfPoints == 0.0:
                    tempArray.append(totalDistance)
                    totalDistance = 0.0
                    medianAcceleration = 0.0
                    tempArray.append(medianAcceleration)
                    totalAcceleration = 0.0

                else:
                    tempArray.append(totalDistance)
                    totalDistance = 0.0
                    medianAcceleration = totalAcceleration / numberOfPoints
                    tempArray.append(medianAcceleration)
                    totalAcceleration = 0.0
                    numberOfPoints = 0.0

            # Speed
            if mobile_records[i][0] == mobile_records[i + 1][0]:
                totaltime = (mobile_records[i][9] - mobile_records[i][8]).total_seconds()
                # print("totaltime is: ", totaltime)
                if mobile_records[i][0] == mobile_records[i + 1][0]:
                    speed = ((((((mobile_records[i][2] / 3.6) + ((mobile_records[i + 1][2] / 3.6))) / 2)) * (
                            mobile_records[i + 1][3] - mobile_records[i][3]).total_seconds()) / totaltime)
                    # print("speed is: ", speed)
                    totalSpeed += speed
            else:
                tempArray.append(totalSpeed)
                tempArray.append(totaltime)
                totalSpeed = 0
                finalArray.append(tempArray)
                tempArray = []
                numberOfTrips = numberOfTrips + 1
                modeArray.append(mobile_records[i][1])
                # finalArray.append(mobile_records[i][1])
                labels.add(mobile_records[i][1])

        labels = list(labels)

        pickle.dump(labels, open("labels.p", "wb"))
        pickle.dump(finalArray, open("X_dataset.p", "wb"))
        pickle.dump(modeArray, open("target.p", "wb"))




    else:

        labels = []
        target = []
        X_dataset = []
        labels = pickle.load(open("labels.p", "rb"))
        target = pickle.load(open("target.p", "rb"))
        X_dataset = pickle.load(open("X_dataset.p", "rb"))

        # encoding the taret array
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        # encoding the target array ( modes int to string)
        encoded_target = le.fit_transform(target)
        print(encoded_labels)

        print(X_dataset)
        print(encoded_target)

        svc_model = LinearSVC(random_state=0)
        X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, encoded_target, test_size=0.30, random_state=10)
        svc_model.fit(X_train, Y_train)
        print(X_test)
        pred = svc_model.predict(X_test)
        print("accuracy : ", accuracy_score(Y_test, pred, normalize=True))