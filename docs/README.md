# pathsight-ai

Specification of the application

- This application is for the oil and gas well planning optimization. The overall idea is that there will be a database where every surface point (x,y) coordinate has a distribution (something like gaussian) that has the mean and variance of the reservoir thickness (in metre) found at that location. There can be more than two distributions for each (x,y) because during the drilling, there are more than two geological units encountered for each depth interval - which means different distribution is warranted.  For different locations (x,y) , the depth interval for the geological unit may not be the same. Not to mention that the mean and variance may not be the same as well because these property can be spatially variant. The main idea is that the user will provide the target (x,y) location for the propose wellhead platform where the well will be started to drill from. We will let the user select the distribuitions via (x,y) on a map (these distribution, of course, comes from the previously drilled wells). The user can select more than two (x,y). Intuitively, they will select those (x,y) that are near to the target (x,y) locaiton. But we will let users do that as they see fit. Once distributions are selected, the user can click a run button and the optimized well path will appear on a map.


- The optimization algorithms will be handled on the background. You don't have to know how it works for now. Your task is to build to user-interface (UI) that facillitates the input from user to our algorithms. To illustrate the functionality, you can use made-up distributions and (x,y) locations as you see fit. You can also generate stright-line well path i.e., (x=constant, y=constant, z=0..3000m with sampling rate of 1m) which represent the vertical well to show the functionality.

- the application will be written fully in Python

- the application will provide graphical user interface (GUI) where the visualization is at the left hand-side and the user-provided options ( (x,y) of wellhead platform, selected distributions, run button).

- the visualization will be a blank space (x,y,z) to show related objects: the well path (x,y,z), a point (x, y, z=0) in a map represent the clickable object that once clicked, the distributions are selected and will be used in the calculation for well path optimization.

- on the right-hand side, there will be a textbox for user to input the surface location (x,y, z=0) for the wellhead platform. User will provide (x,y). Once entered and clicked ok, the wellhead object point should be shown in the left hand-side visualization.

- Furthermore, on the right handside, for the (x,y,z=0) distributions that user select, it will show the distribution data (mean, variance) for geological units (mock up as '2E', '2D' for now).
