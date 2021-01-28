# take all .wav files in curr dir and batch process to mp3 to ./mp3s/ 
# run this shell script after the stimuli are properly cropped to 25 beats 
for file in *.wav; do
dir='./mp3s/';
sox ${file} ${dir}${file%.wav}.mp3;
done; 
