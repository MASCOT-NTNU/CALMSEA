## CALMSEA
We use Expected Improvement to find high densities of Copepod or Chlorophyll. This code was tested in Mausund in June 2024.

### Silcam 
The detection of the Copepods is done using a Silcam. 
The SilCam is a backlit camera installed in the hull of the AUV. When the AUV moves water flows through the opening and the camera can picture what is inside the volume. The images are segmented into individual particles and the paricles are classified
using [PySilCam](https://github.com/SINTEF/PySilCam) developed by SINTEF. 


![D20240605T150920 817540 h5_PN8_p=p=0 9987](https://github.com/MASCOT-NTNU/CALMSEA/assets/56491067/0184bcc2-169c-40f2-8c97-8edd1bf0bd08)![D20240605T104438 248848 h5_PN8_p=p=0 9994](https://github.com/MASCOT-NTNU/CALMSEA/assets/56491067/14519141-14dc-4fc9-a946-c7307d9ff29d)

*Particles classified as a copepod*
