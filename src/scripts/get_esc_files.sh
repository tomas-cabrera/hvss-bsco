p="/home/tomas/cmu/research/hvss-bsco/src/data/cmc"
f="initial.esc.dat"
for c in `ls $p`
do
    echo $c
    c2=`echo $c | perl -pe "s/_/_v2_/"`
    # Change $c/$c2 in the following line as appropriate
    pr="/hildafs/projects/phy200025p/share/catalog_files/$c2.tar.gz/$f"
    pl="$p/$c/$f"
    sshpass -p "PASSWORD" scp tcabrera@data.vera.psc.edu:$pr $pl
done
