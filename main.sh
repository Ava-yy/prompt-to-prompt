for i in {0..10}
do
    seed_start=$((i * 5))
    seed_end=$((i * 5 + 4))
    python main.py $seed_start $seed_end
done
