#!/bin/bash
if [ -f ../user_data/tmp_data/lgb_mae_seed2022.csv ]
then
     echo "The lgb_mae_seed2022 exist"
else
     echo "run lgb_test"
     python lgb_test.py
fi
sleep 2
if [ -f ../user_data/tmp_data/linear.csv ]
then
     echo "The linear exist"
else
     echo "run linear_test"
     python linear_test.py
fi
sleep 2
if [ -f ../prediction_result/sub.csv ]
then
     echo "The sub exist"
else
     echo "run sub_data"
     python sub_data.py
fi
sleep 2