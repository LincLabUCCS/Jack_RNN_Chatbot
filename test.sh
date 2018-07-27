#! /bin/bash
COUNT=1

# notes: 
# cat net-qa.dat | awk -F',' {$1=""; print $0}'

# >&2 echo "net-qa"
# python dialog.py  --loss_model NET --count $COUNT --test_mode QA	 	> net-qa.dat
# >&2 echo "net-1chat"
# python dialog.py  --loss_model NET --count $COUNT --test_mode 1CHAT		> net-1chat.dat
# >&2 echo "net-2chat"
# python dialog.py  --loss_model NET --count $COUNT --test_mode 2CHAT		> net-2chat.dat

>&2 echo "mmi-qa"
python dialog.py  --loss_model MMI --count $COUNT --test_mode QA 		> mmi-qa.dat
>&2 echo "nmi-1chat"
python dialog.py  --loss_model MMI --count $COUNT --test_mode 1CHAT 	> nmi-1chat.dat
>&2 echo "nmi-2chat"
python dialog.py  --loss_model MMI --count $COUNT --test_mode 2CHAT 	> nmi-2chat.dat

>&2 echo "norm-qa"
python dialog.py  --loss_model NORM --count $COUNT --test_mode QA 		> norm-qa.dat
>&2 echo "norm-1chat"
python dialog.py  --loss_model NORM --count $COUNT --test_mode 1CHAT 	> norm-1chat.dat
>&2 echo "norm-2chat"
python dialog.py  --loss_model NORM --count $COUNT --test_mode 2CHAT 	> norm-2chat.dat

>&2 echo "ent-qa"
python dialog.py  --loss_model ENT --count $COUNT --test_mode QA 		> ent-qa.dat
>&2 echo "ent-1chat"
python dialog.py  --loss_model ENT --count $COUNT --test_mode 1CHAT 	> ent-1chat.dat
>&2 echo "ent-2chat"
python dialog.py  --loss_model ENT --count $COUNT --test_mode 2CHAT 	> ent-2chat.dat
