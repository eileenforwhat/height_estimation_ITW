# height_estimation_ITW

## Method
### Person (easiest)
1. Input: two images of a person holding prior object (known height, same plane as person)
2. Extract keypoints and match correspondences (SIFT)
3. Compute fundamental matrix F and camera matrices {P1, P2}
4. Triangulate using manual annotations of person + prior object
5. Calibrate projective ambiguity 
6. Height estimation
### Building (harder)
### Stretch: mountain (hardest)

## Concerns
- do we need to annotate the base? if we don't 
- should we assume that the image does not have any roll? 
- how do we use the prior for scaling in the existence of projective ambiguity? 
    - if the prior and the tall object are both on the same plane it shouldn't matter. 
    - In the case of a person holding a cereal box, we can safely assume that the prior and the peak are on the same plane
    - In the case of a tall building/mountain, our prior is the GPS coordinates hmmmm they're not on the same plane
    - can be fixed with metric rectification
- should we perform metric rectification on it? 
    - not for the 2 view geometry case. let's perform dumb scaling in all three directions for now.
- perform ground-plane calibration from three collinear points (ref: https://www.researchgate.net/figure/Plant-height-measurements-using-stereo-vision_fig1_265797007)


## To Do: two-view reconstruction
- [ ] SIFT for keypoint detection & correspoondence - eileen
- [x] copy F computation from assignments - simon
- [ ] compute P, P' from F - simon


## To Do: SfM
- [ ] Item 1
- [ ] Item 2


## timeline

Work session (Thursday 11/30)
1. two view reconstruction on human body, figure out the kinks

Work session (Saturday 12/2)
2. collect photos of tall building [cathedral of learning]
3. two view reconstruction on building, should be easy
4. multi-view sfm on tall building, need to implement

Work session (put together results week of presentation: TBD date)
7. presentation (Thursday, December 7, 2023)

Extensions (maybe) after presentation:
- think about single-view case.
- mountains


Some other common deadlines
- 12/7 geoviz presentation
- 12/8 comphoto hw6 due
- 12/8 capstone poster session
- 12/14 geoviz project reports due
- 12/18 comphoto project due

