DIRBAK=$(pwd)
#TGTDIR="../youth-mental-health-runtime/submission_src"
TGTDIR="../data/submission"


echo $TGTDIR
echo $1
echo "=========step $2"

# search thr
python search_model_weight.py "$1" $2
python search_thr.py "$1" "$2"


# prepare
rm -rf $TGTDIR/*
mkdir -p $TGTDIR/yaa/data
mkdir -p $TGTDIR/yaa/yaa

# thr
cp ../data/thrs.dump $TGTDIR/yaa/data
cp ../data/model_weights.dump $TGTDIR/yaa/data

# code
cp main.py $TGTDIR
cp eval.py $TGTDIR/yaa/yaa
cp main.py $TGTDIR/yaa/yaa
cp util.py $TGTDIR/yaa/yaa
cp dataset.py $TGTDIR/yaa/yaa
cp trainer.py $TGTDIR/yaa/yaa
cp nn.py $TGTDIR/yaa/yaa
cp ppts.py $TGTDIR/yaa/yaa
#cp -r ../unsloth-zoo/unsloth_zoo $TGTDIR/yaa/
#cp -r ../unsloth_code/unsloth $TGTDIR/yaa/
#cp -r ../trl_code/trl $TGTDIR/yaa/

# lora
mkdir -p $TGTDIR/yaa/data/hfmodels/unsloth
python fix_lora.py "$1"

# model
cd $TGTDIR/yaa/data
for modelname in $1
do
  ln -s "../../../$modelname" "$modelname"
done

# pack
cd $DIRBAK
cd $TGTDIR
zip -r submission.zip ./* -x ./yaa/data/*/events* -x ./yaa/data/*/pred*

cd $DIRBAK
#cd ../youth-mental-health-runtime
#mkdir -p submission/
#cd submission_src; rm -rf ../submission/submission.zip; zip -r ../submission/submission.zip ./*
#cd $DIRBAK
#cd ../youth-mental-health-runtime
#sudo make test-submission

cd $DIRBAK

