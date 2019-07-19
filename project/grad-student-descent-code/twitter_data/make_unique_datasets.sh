awk '{$1=$1}1' train_neg_full.txt | awk ' !x[$0]++' | tee train_neg_full_uniq.txt

awk '{$1=$1}1' train_pos_full.txt | awk ' !x[$0]++' | tee train_pos_full_uniq.txt

awk '{$1=$1}1' train_neg.txt | awk ' !x[$0]++' | tee train_neg_uniq.txt

awk '{$1=$1}1' train_pos.txt | awk ' !x[$0]++' | tee train_pos_uniq.txt