# Change Detection Course Project
Goal of the project is to detect changes between two frames of images(query and reference). 
* General Process:
    * Segment the query image
    * Extract features/descriptors of query and reference image
    * Compare images via feature descriptors
    * Construct mask containing regions of the query image with low feature percent match
    * Return image containing only the regions that have changed.

The implemented process is shown below:
![Alt text](resources/ChangeDetectionProcess.png?raw=true "Change Detection Process")

