for f in *.mp4
do
 ffmpeg -i $f -ab 1k -ac 1 -ar 44100 -vn ${f/.mp4/.wav}
done