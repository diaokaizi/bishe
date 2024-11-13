import sys 
import os
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_dir)
from sklearn.svm import OneClassSVM
import report_result
import read_data

model_name = "ocsvm"


# (x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
# filepath = "load_cic2018_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
# filepath = "load_UGR16_faac"
(x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
filepath = "load_cic2017_faac"

iof = OneClassSVM()
iof=iof.fit(x_train)
score=-iof.decision_function(x_test) #值越低越不正常

report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test)