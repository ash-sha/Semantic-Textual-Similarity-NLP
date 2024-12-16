                       BioCreative/OHNLP Challenge 2018 Task 2
                    Clinical Semantic Textual Similarity (ClinicalSTS) 

This package contains the testing data for the Clinical Semantic Textual Similarity
(ClinicalSTS) shared task. The dataset has the following tab-separated format:

  * One STS pair per line.
  * Each line contains the following fields: STS Sent1[tab]STS Sent2


-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Submission Format: 
-------------
For the testing data, please generate a plain text output file of the form
ClinicalSTS2018.OUTPUT.$TEAM_NAME.$RunName.txt

Example: ClinicalSTS2018.OUTPUT.MyTeam.LSTM-Run.txt

The evaluation file should have one line for each STS pair that provides the 
score assigned by your system as a floating point number:

0.1
4.9
3.5
2.0

By default, each team can submit up to three runs. Teams that feel they have a 
good reason for submitting more than three runs should contact the organizers.
Such requests will be handled on a case by case basis.

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Submission Guideline: 
-------------
Zip your runs in one .zip file and submit to https://easychair.org/conferences/?conf=clinicalsts2018

Please provide a short description of each run in the abstract. Include a list of any external resources you may have used in the run. 

Good Luck!
