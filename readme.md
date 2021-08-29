# SmartCue
### A rewrite of my [PoolPredictor](https://github.com/TheDitis/PoolPredictor) project
#### This readme is largely coppied from my original PoolPredictor readme. That project is farther along, but of much, much lower code quality. I will update once this project is further along.

### Example From Old Version (PoolPredictor):
![Demo Gif](https://github.com/TheDitis/PoolPredictor/blob/master/doc_resources/PoolGif1.gif)

<br/><br/>
# The Goal:
### To be able to mount a camera and projector over a pool table, and project lines of motion. The hope is to identify when a que is lined up, so that a forecasted line of motion can be projected for the queball in realtime, showing which balls or walls might be hit, and draw any trajectory lines for those balls.







<br/><br/>
# The Algorithm:
### 1. Take a few frames, and identify long straight lines throughout them via Hough Transform, filtering them based on relative distance and orientation relationships.
### 2. Separate found lines into bumper, pocket, and table edge lines.
### 3. Identify pocket areas based on line intersections and lengths between intersections.
### 4. Start checking frames live, looking only within the table boundaries for circles.
### 5. Keep only circles with radii within plausable range for a billiard ball
### 6. Check the average color within each circle against that of the table cloth. if it's within a certain threshold of similarity, throw it out.
## NOTE: SmartCue is not yet past step 6. Further steps apply only to PoolPredictor (old version)
### 7. It keeps a fixed length FIFO buffer of frames (currently 5) where ball qualified objects are stored.
### 8. With each new frame each new circle in that frame is checked against the balls objects in the frame buffer for color-similarity, and distance. If a good match is found, the circle will be added to the frame buffer for that ball.
### 9. If circles being added to a ball start having regular distance from eachother then that ball enters a 'moving' state
### 10. A line is fit to the centers of circles in the moving balls buffer, and if the line fits nicely with little deviation, then direction of motion, as well as velocity is assessed.
### 11. A peliminary velocity vector is instatiated, and it is analyzed for bumper intersections
### 12. If an intersection is detected, a reflection vector is made.
### 13. Collision detection is done by checking all balls distance from any lines of motion. If the distance is within 2r, it flashes red.
