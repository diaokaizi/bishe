import sys 
import os
import time
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_dir)
from sklearn.svm import OneClassSVM
import report_result
import read_data

model_name = "ocsvm"

def UGR16():

    (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
    filepath = "load_UGR16_faac"

    iof = OneClassSVM(nu=0.000001)
    iof=iof.fit(x_train)
    start_time = time.time()
    score=-iof.decision_function(x_test) #值越低越不正常
    end_time = time.time()
    testing_time_cost = end_time - start_time
    print(f"Testing Time Cost: {testing_time_cost * 1000} seconds")
    f1, resullt_str = report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test) 

def cic2017():
    (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
    filepath = "load_cic2017_faac"

    iof = OneClassSVM(gamma=50)
    iof=iof.fit(x_train)
    start_time = time.time()
    score=-iof.decision_function(x_test) #值越低越不正常
    end_time = time.time()
    testing_time_cost = end_time - start_time
    print(f"Testing Time Cost: {testing_time_cost * 1000} seconds")
    f1, resullt_str = report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test) 


def cic2018():
    (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
    filepath = "load_cic2018_faac"

    iof = OneClassSVM(gamma=50)
    iof=iof.fit(x_train)
    start_time = time.time()
    score=-iof.decision_function(x_test) #值越低越不正常
    end_time = time.time()
    testing_time_cost = end_time - start_time
    print(f"Testing Time Cost: {testing_time_cost * 1000} seconds")
    f1, resullt_str = report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test) 


if __name__ == "__main__":
    UGR16()
    cic2017()
    cic2018()