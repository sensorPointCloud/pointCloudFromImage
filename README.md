**Install**
- Tested with python 2.7x, pptk currently not working with 2.8+
- run: pip install -r requirements.txt

# USAGE
    # check that all tests pass
        # go to project directory
        cd tests
        pytest

    # generate results
        # go to project directory
        cd results/fine_step_z_270_deg
        python generate_point_cloud.py

    # view point cloud
        # go to project directory
            cd results/fine_step_z_270_deg
            python view_point_cloud.py