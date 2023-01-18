<h1>Emotion Recognition</h1>
The project scope is emotion detection, training on recorded data and recognition. All files and back-up copies of recorded data are preserved<br>
Files work in the following way:<br>
<br><b>1. Capture.py:</b> Capture facial data and corresponding facial coordinates
<br><b>2. Train.py: </b> Training over recorded data
<br><b>3. Detect.py: </b> Loading trained model and recognition process
<br><br>If you want to add your own facial landmarks and/or change used model - you need to go though all steps.
<br>If you want to use already pretrained model and recorded facial data - just run detect.py and enjoy ;)
<br>Additionaly, there are 3 .csv files - coords, coords_base and coords_orig. <b>Coords_base</b> is just set of x, y, z coordinates in case you want to start recording a new - may save you some works. <b>Coords</b> is currently working file and <b>Coords_orig</b> is it's back-up. 
