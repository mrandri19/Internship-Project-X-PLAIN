```
(lace) ➜  Demo_LACE_v0 git:(master) ✗ python profiling.py
readlines running time: 0.12336421012878418
domain running time: 0.12350964546203613
data_to_instance running time: 0.7979986667633057
table running time: 1.5892226696014404
loadARFF_Weka running time: 2.510845899581909
readlines running time: 0.14403510093688965
domain running time: 0.14416003227233887
data_to_instance running time: 0.8793730735778809
table running time: 1.881702184677124loadARFF_Weka running time: 2.905374765396118
explainer creation time: 5.929206371307373

computePredictionDifferenceSubsetRandomOnlyExisting
[[1, 7, 9], [1, 9, 10], [4, 9, 10], [1, 4, 7, 9, 10]]computePredictionDifferenceSinglever2 time: 7.702129125595093

computePredictionDifferenceSubsetRandomOnlyExisting
[[1, 8, 9, 10], [1, 2, 7, 8, 9], [1, 2, 9, 10], [4, 10], [1, 2, 4, 7, 8, 9, 10]]
computePredictionDifferenceSinglever2 time: 27.004581451416016

computePredictionDifferenceSubsetRandomOnlyExisting
[[1, 8, 9, 10], [1, 2, 9, 10], [4, 10], [1, 4, 9], [1, 6, 7, 8, 9], [1, 2, 7, 8, 9, 11], [1, 2, 4, 6, 7,
8, 9, 10, 11]]
computePredictionDifferenceSinglever2 time: 100.4552960395813
computePredictionDifferenceSubsetRandomOnlyExisting
[[1, 2, 9, 10], [2, 4, 8, 10], [1, 2, 4, 9], [1, 6, 7, 8, 9], [1, 2, 7, 8, 9, 11], [1, 2, 4, 6, 7, 8, 9,
10, 11]]computePredictionDifferenceSinglever2 time: 107.3817687034607

computePredictionDifferenceSubsetRandomOnlyExisting
[[1, 2, 4], [2, 4, 8, 10], [1, 6, 7, 8, 9], [1, 2, 4, 6, 7, 8, 9, 10]]
computePredictionDifferenceSinglever2 time: 54.95623230934143

Local rules:
Rule_1  ->  {sex=Female, workclass=Private, race=Black}
Rule_2  ->  {workclass=Private, race=Black, capital-gain=low, hours-per-week=39.5-40.5}
Rule_3  ->  {sex=Female, occupation=Blue-Collar, relationship=Unmarried, capital-gain=low, capital-loss=l
ow}
Union of rule bodies:
Rule_U  ->  {sex=Female, workclass=Private, race=Black, occupation=Blue-Collar, relationship=Unmarried, c
apital-gain=low, capital-loss=low, hours-per-week=39.5-40.5}
explanation time: 320.4526288509369
```
