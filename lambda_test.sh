## Test on Lambda_ce

for l_ce in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do

python main.py --lambda_ce $l_ce --lambda_cr 0.5

done

## Test on Lambda_cr

for l_cr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do

python main.py --lambda_cr $l_cr --lambda_ce 0.5

done
