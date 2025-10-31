

nohup python -u download_data.py 0 100 > a.log 2>&1 &
sleep 10
nohup python -u download_data.py 101 200 > b.log 2>&1 &
sleep 10
nohup python -u download_data.py 201 300 > c.log 2>&1 &
sleep 10
nohup python -u download_data.py 301 400 > d.log 2>&1 &
sleep 10
nohup python -u download_data.py 401 500 > e.log 2>&1 &
sleep 10
nohup python -u download_data.py 501 600 > f.log 2>&1 &
sleep 10
nohup python -u download_data.py 601 700 > g.log 2>&1 &