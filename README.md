deeptest
========

+ Gen code for running on grid
overfeat
>> gen_sge_code('overfeat_predict_sge', 'youcook %d %d', 88, 28, 4)
>> gen_sge_code('overfeat_predict_sge', 'youtube2text %d %d', 1970, 28, 4)
>> gen_sge_code('overfeat_predict_sge', 'medresearch %d %d', 10161, 28, 4)

deepcafee
>> gen_sge_code('deepcaffe_predict_sge', 'youcook %d %d', 88, 50, 4)
>> gen_sge_code('deepcaffe_predict_sge', 'youtube2text %d %d', 1970, 50, 4)
>> gen_sge_code('deepcaffe_predict_sge', 'medresearch %d %d', 10161, 50, 4)


deepcafee (predict scores)
gen_sge_code('deepcaffe_predict_scores_sge', 'trecvidmed10 devel %d %d', 1744, 28, 4)
gen_sge_code('deepcaffe_predict_scores_sge', 'trecvidmed10 test %d %d', 1724, 28, 4)
gen_sge_code('deepcaffe_predict_scores_sge', 'trecvidmed11 devel %d %d', 11783, 28, 4)
gen_sge_code('deepcaffe_predict_scores_sge', 'trecvidmed11 test %d %d', 31999, 28, 4)

Note: 
server bc3 has 136 cores (8x8 + 6x12)
server bc4 has 132 cores (6x8 + 7x12)
bc3 + bc4 = 268 cores
>> gen_sge_code('deepcaffe_predict_med', 'kindredtest14 %d %d', 14708, 67, 4)
>> gen_sge_code('deepcaffe_predict_med', 'medtest14 %d %d', 27275, 67, 4)    
>> gen_sge_code('deepcaffe_predict_med', 'event %d %d', 6964, 67, 4)


gen_sge_code('deepcaffe_predict_med_fc7', 'kindredtest14 %d %d', 14708, 65, 4, 'kindredtest14.')
gen_sge_code('deepcaffe_predict_med_fc7', 'medtest14 %d %d', 27275, 65, 4, 'medtest14.')    
gen_sge_code('deepcaffe_predict_med_fc7', 'event %d %d', 6964, 65, 4, 'event.')
gen_sge_code('deepcaffe_predict_med_fc7', 'eventbg %d %d', 4992, 65, 4, 'eventbg.')
