import sys 
import os
root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_dir)
from sklearn.ensemble import IsolationForest
import report_result
import read_data
import time
model_name = "iForest"

# (x_train, y_train), (x_test, y_test) = read_data.load_UGR16_faac()
# filepath = "load_UGR16_faac"
# (x_train, y_train), (x_test, y_test) = read_data.load_cic2017_faac()
# filepath = "load_cic2017_faac"
(x_train, y_train), (x_test, y_test) = read_data.load_cic2018_faac()
filepath = "load_cic2018_faac"
iof = IsolationForest()
iof=iof.fit(x_train)
start_time = time.time()
score=-iof.decision_function(x_test) #值越低越不正常
end_time = time.time()
testing_time_cost = end_time - start_time
print(f"Testing Time Cost: {testing_time_cost * 1000} seconds")
report_result.report_result(model=model_name, name=filepath, anomaly_score=score, labels=y_test)