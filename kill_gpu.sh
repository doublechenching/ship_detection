state=$(sudo fuser -v /dev/nvidia$1)
i=0
for pid in $state
do 
    if [ $i -eq 0 ]
    then
        i=1
        continue
    fi
    echo "killing $pid"
    kill $pid
done