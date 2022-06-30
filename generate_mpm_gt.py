


# For both sequences of Fluo-N2DL-HeLa
    # load frames into napari

    # segment with stardist
    # ok so lowkey maybe it's not good enough lmao we need to either crop or validate.... smh
            # let's crop and validate...

    # take labels layer, compute centers of mass (with overlaps)

    # for each point coord per frame belonging to the same cell ID

        # write out to sample_gt.txt: frame label_val point_coord_x point_coord_y parent (only if it has split, otherwise it's just -1)
            # who's generating all these children...?
            # parent is just pixel value that spawned these children - can we determine that...?
                # given an ID we've never seen before, look for IDs in pixel neighbourhood in previous frame and find that which is no longer present in the current frame
                    # expensive? Yes but w.e.? 

                # then just do a quick pass over the end and remove any IDs that occur in just one frame?



