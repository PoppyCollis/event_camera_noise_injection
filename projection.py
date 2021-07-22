
from datetime import datetime
from multiprocessing import Process
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import tables
import stl
import cv2
import dv


def create_labelled_data_file(f_name='./data/labelled.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))
    f_label = f.create_earray(f.root, 'label', tables.atom.UInt8Atom(), (0,))
    f.close()


def get_dv_events(address, port, record_time, camera_matrix, distortion_coefficients,
                  f_name='./data/dv_event.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))

    with dv.NetworkEventInput(address=address, port=port) as event_f:
        event = next(event_f)
        stop_time = event.timestamp + record_time

        for event in event_f:
            if event.timestamp >= stop_time:
                break

            # undistort event
            event_distorted = np.array([event.x, event.y], dtype='float64')
            event_undistorted = cv2.undistortPoints(
                event_distorted, camera_matrix, distortion_coefficients, None, camera_matrix)[0, 0]

            f_timestamp.append([event.timestamp])
            f_polarity.append([event.polarity])
            f_x.append([event_undistorted[0]])
            f_y.append([event_undistorted[1]])

    f.close()
    return


def get_dv_frames(address, port, record_time, camera_matrix, distortion_coefficients,
                  f_name='./data/dv_frame.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp_a = f.create_earray(f.root, 'timestamp_a', tables.atom.UInt64Atom(), (0,))
    f_timestamp_b = f.create_earray(f.root, 'timestamp_b', tables.atom.UInt64Atom(), (0,))
    f_image = f.create_earray(f.root, 'image', tables.atom.UInt8Atom(), (0, 260, 346, 3))

    with dv.NetworkFrameInput(address=address, port=port) as frame_f:
        frame = next(frame_f)
        stop_time = frame.timestamp_end_of_frame + record_time

        for frame in frame_f:
            if frame.timestamp_end_of_frame >= stop_time:
                break

            # undistort frame
            frame_distorted = frame.image[:, :, :]
            frame_undistorted = cv2.undistort(
                frame_distorted, camera_matrix, distortion_coefficients, None, camera_matrix)

            f_timestamp_a.append([frame.timestamp_start_of_frame])
            f_timestamp_b.append([frame.timestamp_end_of_frame])
            f_image.append([frame_undistorted])

    f.close()
    return


def get_vicon_poses_pyvicon(address, port, record_time, prop_mesh_markers,
                            f_name='./data/vicon.h5'):

    import pyvicon as pv

    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in prop_mesh_markers.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'quality', tables.atom.Float64Atom(), (0,))
        f.create_earray(g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in prop_mesh_markers[prop_name].keys():
            f.create_earray(g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    client = pv.PyVicon()
    #print('version: ' + client.__version__)

    result = client.connect(f'{address}:{port}')
    #print('connect:', result)

    result = client.enable_marker_data()
    #print('enable_marker_data:', result)

    result = client.enable_segment_data()
    #print('enable_segment_data:', result)


    sanity_check = False

    if sanity_check:
        while True:
            result = client.get_frame()
            print('get_frame:', result)

            if result != pv.Result.NoFrame:
                break

        prop_count = client.get_subject_count()
        print('prop count:', prop_count)
        assert(prop_count == len(prop_mesh_markers))

        for prop_i in range(prop_count):
            prop_name = client.get_subject_name(prop_i)
            print('prop name:', prop_name)
            assert(prop_name in prop_mesh_markers.keys())

            marker_count = client.get_marker_count(prop_name)
            print(' ', prop_name, 'marker count:', marker_count)
            assert(marker_count == len(prop_mesh_markers[prop_name]))

            for marker_i in range(marker_count):
                marker_name = client.get_marker_name(prop_name, marker_i)
                print('   ', prop_name, 'marker', marker_i, 'name:', marker_name)
                assert(marker_name in prop_mesh_markers[prop_name].keys())


    timestamp = int(datetime.now().timestamp() * 1000000)
    stop_time = timestamp + record_time

    while timestamp < stop_time:
        result = client.get_frame()

        if result == pv.Result.NoFrame:
            continue

        timestamp = int(datetime.now().timestamp() * 1000000)
        f_timestamp.append([timestamp])

        prop_count = client.get_subject_count()

        for prop_i in range(prop_count):
            prop_name = client.get_subject_name(prop_i)            

            marker_count = client.get_marker_count(prop_name)

            prop_quality = client.get_subject_quality(prop_name)
            g_props[prop_name].quality.append([prop_quality])

            if prop_quality is not None:
                root_segment = client.get_subject_root_segment_name(prop_name)

                rotation = client.get_segment_global_rotation_euler_xyz(prop_name, root_segment)
                g_props[prop_name].rotation.append([rotation])

                for marker_i in range(marker_count):
                    marker_name = client.get_marker_name(prop_name, marker_i)

                    translation = client.get_marker_global_translation(prop_name, marker_name)
                    g_props[prop_name].translation[marker_name].append([translation])

            else:
                rotation = np.full((1, 3), np.nan)
                g_props[prop_name].rotation.append(rotation)

                for marker_i in range(marker_count):
                    marker_name = client.get_marker_name(prop_name, marker_i)

                    translation = np.full((1, 3), np.nan)
                    g_props[prop_name].translation[marker_name].append(translation)

    result = client.disconnect()
    # #print('disconnect:', result)

    f.close()
    return


def get_vicon_poses(address, port, record_time, prop_mesh_markers,
                    f_name='./data/vicon.h5'):

    from vicon_dssdk import ViconDataStream

    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in prop_mesh_markers.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'quality', tables.atom.Float64Atom(), (0,))
        f.create_earray(g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in prop_mesh_markers[prop_name].keys():
            f.create_earray(g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    client = ViconDataStream.Client()
    #print('version: ' + str(client.GetVersion()))
    client.Connect(f'{address}:{port}')
    client.EnableMarkerData()
    client.EnableSegmentData()


    sanity_check = False

    if sanity_check:
        while True:
            if client.GetFrame():
                break

        prop_names = client.GetSubjectNames()
        prop_count = len(prop_names)
        print('prop count:', prop_count)
        assert(prop_count == len(prop_mesh_markers))

        for prop_i in range(prop_count):
            prop_name = prop_names[prop_i]
            print('prop name:', prop_name)
            assert(prop_name in prop_mesh_markers.keys())

            marker_names = client.GetMarkerNames(prop_name)
            marker_count = len(marker_names)
            print(' ', prop_name, 'marker count:', marker_count)
            assert(marker_count == len(prop_mesh_markers[prop_name]))

            for marker_i in range(marker_count):
                marker_name = marker_names[marker_i][0]
                print('   ', prop_name, 'marker', marker_i, 'name:', marker_name)
                assert(marker_name in prop_mesh_markers[prop_name].keys())


    timestamp = int(datetime.now().timestamp() * 1000000)
    stop_time = timestamp + record_time

    while timestamp < stop_time:
        if not client.GetFrame():
            continue

        timestamp = int(datetime.now().timestamp() * 1000000)
        f_timestamp.append([timestamp])

        prop_names = client.GetSubjectNames()
        prop_count = len(prop_names)

        for prop_i in range(prop_count):
            prop_name = prop_names[prop_i]            

            marker_names = client.GetMarkerNames(prop_name)
            marker_count = len(marker_names)

            try:
                prop_quality = client.GetObjectQuality(prop_name)
            except ViconDataStream.DataStreamException:
                prop_quality = None
            g_props[prop_name].quality.append([prop_quality])

            if prop_quality is not None:
                root_segment = client.GetSubjectRootSegmentName(prop_name)

                rotation = client.GetSegmentGlobalRotationEulerXYZ(prop_name, root_segment)[0]
                g_props[prop_name].rotation.append([rotation])

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                    g_props[prop_name].translation[marker_name].append([translation])

            else:
                rotation = np.full((1, 3), np.nan)
                g_props[prop_name].rotation.append(rotation)

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = np.full((1, 3), np.nan)
                    g_props[prop_name].translation[marker_name].append(translation)

    client.Disconnect()

    f.close()
    return


def projection():

    prop_mesh = stl.mesh.Mesh.from_file('./screwdriver-decimated.stl')

    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address, dv_event_port, dv_frame_port = '127.0.0.1', 36000, 36001

    dv_space_coefficients_file = './calibration/dv_space_coefficients.npy'
    dv_space_constants_file = './calibration/dv_space_constants.npy'

    dv_space_coefficients = np.load(dv_space_coefficients_file)
    dv_space_constants = np.load(dv_space_constants_file)

    prop_mesh_markers = {}

    # # screwdriver mesh marker coordinates
    # prop_mesh_markers['jt_screwdriver'] = {
    #     'handle_1':    [ 0.0,  78.0,   13.5],
    #     'handle_2':    [ 0.0,  78.0,  -13.5],
    #     'shaft_base':  [ 5.0,  120.0,  0.0 ],
    #     'shaft_tip':   [-5.0,  164.0,  0.0 ],
    # }

    # PROTOTYPE: screwdriver mesh marker coordinates
    prop_mesh_markers['jt_screwdriver'] = {
        'handle_1':    [ 0.0,  78.0,   13.5],
        'handle_2':    [ 0.0,  78.0,  -13.5],
        'shaft_base':  [ 7.5,  100.0,  0.0 ], # alternate position
        'shaft_tip':   [-5.0,  164.0,  0.0 ],
    }

    # Vicon wand mesh marker coordinates
    prop_mesh_markers['jt_wand'] = {
        'top_left':    [ 0.0,  0.0,  0.0 ],
        'top_centre':  [ 0.0,  0.0,  0.0 ],
        'top_right':   [ 0.0,  0.0,  0.0 ],
        'middle':      [ 0.0,  0.0,  0.0 ],
        'bottom':      [ 0.0,  0.0,  0.0 ],
    }


    #record = True
    record = False

    test_name = 'noise'

    test_number = 1

    #record_seconds = 3
    record_seconds = 10
    #record_seconds = 100
    record_time = record_seconds * 1000000

    #vicon_usec_offset = 183000
    #vicon_usec_offset = 157000
    #vicon_usec_offset = 69000000
    vicon_usec_offset = -560000  
    #vicon_usec_offset = 0

    distinguish_dv_event_polarity = False



    dv_camera_matrix = np.load('./calibration/camera_matrix.npy')
    dv_distortion_coefficients = np.load('./calibration/camera_distortion_coefficients.npy')

    f_labelled_name = f'./data/labelled_{test_name}_{test_number:04}.h5'
    #f_dv_event_name = f'./data/dv_event_{test_name}_{test_number:04}.h5'
    f_dv_event_name = f'./data/combo_event.h5'

    f_dv_frame_name = f'./data/dv_frame_{test_name}_{test_number:04}.h5'
    f_vicon_name = f'./data/vicon_{test_name}_{test_number:04}.h5'

    f_event_image_video = f'./data/event_image_video_{test_name}_{test_number:04}.avi'
    f_frame_image_video = f'./data/frame_image_video_{test_name}_{test_number:04}.avi'



    ##################################################################


    if record:
        print('=== begin recording ===')

        processes = []
        processes.append(Process(target=get_dv_events,
                                 args=(dv_address, dv_event_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'f_name': f_dv_event_name}))
        processes.append(Process(target=get_dv_frames,
                                 args=(dv_address, dv_frame_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'f_name': f_dv_frame_name}))
        processes.append(Process(target=get_vicon_poses,
                                 args=(vicon_address, vicon_port, record_time,
                                       prop_mesh_markers),
                                 kwargs={'f_name': f_vicon_name}))

        #processes.append(Process(target=get_vicon_poses_pyvicon,
        #                         args=(vicon_address, vicon_port, record_time,
        #                               prop_mesh_markers),
        #                         kwargs={'f_name': f_vicon_name}))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('=== end recording ===')

        exit(0)


    ##################################################################



    create_labelled_data_file(f_name=f_labelled_name)

    f_labelled = tables.open_file(f_labelled_name, mode='a')
    f_labelled_timestamp = f_labelled.root.timestamp
    f_labelled_polarity = f_labelled.root.polarity
    f_labelled_x = f_labelled.root.x
    f_labelled_y = f_labelled.root.y
    f_labelled_label = f_labelled.root.label

    f_dv_event = tables.open_file(f_dv_event_name, mode='r')
    f_dv_event_timestamp = f_dv_event.root.timestamp.iterrows()
    f_dv_event_polarity = f_dv_event.root.polarity.iterrows()
    f_dv_event_x = f_dv_event.root.x.iterrows()
    f_dv_event_y = f_dv_event.root.y.iterrows()

    f_dv_frame = tables.open_file(f_dv_frame_name, mode='r')
    f_dv_frame_timestamp_a = f_dv_frame.root.timestamp_a.iterrows()
    f_dv_frame_timestamp_b = f_dv_frame.root.timestamp_b.iterrows()
    f_dv_frame_image = f_dv_frame.root.image.iterrows()

    f_vicon = tables.open_file(f_vicon_name, mode='r')
    f_vicon_timestamp = f_vicon.root.timestamp.iterrows()
    f_vicon_quality = {}
    f_vicon_rotation = {}
    f_vicon_translation = {}
    for prop in f_vicon.root.props:
        prop_name = prop._v_name
        f_vicon_quality[prop_name] = prop.quality.iterrows()
        f_vicon_rotation[prop_name] = prop.rotation.iterrows()
        f_vicon_translation[prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            f_vicon_translation[prop_name][marker_name] = marker.iterrows()


    def get_next_dv_event(f_timestamp, f_polarity, f_x, f_y, usec_offset=0):
        timestamp = np.uint64(next(f_timestamp) + usec_offset)
        polarity = next(f_polarity)
        x = next(f_x)
        y = next(f_y)

        return timestamp, polarity, x, y


    def get_next_dv_frame(f_timestamp_a, f_timestamp_b, f_image, usec_offset=0):
        timestamp_a = np.uint64(next(f_timestamp_a) + usec_offset)
        timestamp_b = np.uint64(next(f_timestamp_b) + usec_offset)
        image = next(f_image)

        return timestamp_a, timestamp_b, image


    def get_next_vicon(f_timestamp, f_quality, f_rotation, f_translation, usec_offset=0):
        timestamp = np.uint64(next(f_timestamp) + usec_offset)

        quality = {}
        for prop_name in f_quality.keys():
            quality[prop_name] = next(f_quality[prop_name])

        rotation = {}
        for prop_name in f_rotation.keys():
            rotation[prop_name] = next(f_rotation[prop_name])

        translation = {}
        for prop_name in f_translation.keys():
            translation[prop_name] = {}
            for marker_name in f_translation[prop_name].keys():
                translation[prop_name][marker_name] = next(f_translation[prop_name][marker_name])

        return timestamp, quality, rotation, translation


    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    grey = (100, 100, 100)

    dv_frame_shape = (260, 346, 3)
    prop_mask = np.zeros(dv_frame_shape[:2], dtype='uint8')
    event_pos = np.zeros(dv_frame_shape[:2], dtype='uint64')
    event_neg = np.zeros(dv_frame_shape[:2], dtype='uint64')
    event_image = np.zeros(dv_frame_shape, dtype='uint8')
    frame_image = np.zeros(dv_frame_shape, dtype='uint8')

    event_image_video = cv2.VideoWriter(
        f_event_image_video, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_frame_shape[1::-1])
    frame_image_video = cv2.VideoWriter(
        f_frame_image_video, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_frame_shape[1::-1])

    vicon_timestamp, vicon_quality, vicon_rotation, vicon_translation = get_next_vicon(
        f_vicon_timestamp, f_vicon_quality, f_vicon_rotation, f_vicon_translation,
        usec_offset=vicon_usec_offset,
    )

    dv_event_timestamp, dv_event_polarity, dv_event_x, dv_event_y = get_next_dv_event(
        f_dv_event_timestamp, f_dv_event_polarity, f_dv_event_x, f_dv_event_y,
    )

    dv_frame_timestamp_a, dv_frame_timestamp_b, dv_frame_image = get_next_dv_frame(
        f_dv_frame_timestamp_a, f_dv_frame_timestamp_b, f_dv_frame_image,
    )






    # === MAIN LOOP ===
    while True:
        print('vicon timestamp: ', vicon_timestamp, '\tDV frame timestamp: ', dv_frame_timestamp_a)



        # TODO: loop over all props

        prop_name = 'jt_screwdriver'





        # get next Vicon frame
        try:
            vicon_timestamp_new, vicon_quality_new, vicon_rotation_new, vicon_translation_new = get_next_vicon(
                f_vicon_timestamp, f_vicon_quality, f_vicon_rotation, f_vicon_translation,
                usec_offset=vicon_usec_offset,
            )

        except StopIteration:
            break

        # get timestamp halfway between now and the next vicon frame
        vicon_timestamp_midway = (vicon_timestamp + vicon_timestamp_new) / 2
        vicon_timestamp = vicon_timestamp_new
        vicon_quality = vicon_quality_new
        vicon_translation = vicon_translation_new
        vicon_rotation = vicon_rotation_new

        # clear prop mask
        prop_mask.fill(0)

        # get mesh and vicon marker translations for this prop
        x = np.array([translation for translation in prop_mesh_markers[prop_name].values()])
        y = np.array([translation for translation in vicon_translation[prop_name].values()])

        if np.isfinite(x).all() and np.isfinite(y).all():
            # estimate Vicon space transformation
            regressor = MultiOutputRegressor(
                estimator=LinearRegression(),
            ).fit(x, y)

            vicon_space_coefficients = np.array([re.coef_ for re in regressor.estimators_]).T
            vicon_space_constants = np.array([[re.intercept_ for re in regressor.estimators_]])

            # transform STL mesh space to Vicon space, then
            # transform from Vicon space to DV camera space
            vicon_space_p = np.matmul(prop_mesh.vectors, vicon_space_coefficients) + vicon_space_constants
            dv_space_p = np.matmul(vicon_space_p, dv_space_coefficients) + dv_space_constants
            dv_space_p = dv_space_p[:, :, :2] * (1.0 / dv_space_p[:, :, 2, np.newaxis])
            dv_space_p_int = np.rint(dv_space_p).astype('int32')

            # compute prop mask
            cv2.fillPoly(prop_mask, dv_space_p_int, 255)
            #prop_mask_dilation_kernel = np.ones((3, 3), 'uint8')
            prop_mask_dilation_kernel = np.ones((6, 6), 'uint8')            
            prop_mask = cv2.dilate(prop_mask, prop_mask_dilation_kernel)


        # get next DV events
        event_pos.fill(0)
        event_neg.fill(0)

        try:
            while dv_event_timestamp < vicon_timestamp_midway:
                bounded_x = 0 <= dv_event_x < dv_frame_shape[1]
                bounded_y = 0 <= dv_event_y < dv_frame_shape[0]

                if bounded_x and bounded_y:
                    if dv_event_polarity:
                        event_pos[dv_event_y, dv_event_x] += 1
                    else:
                        event_neg[dv_event_y, dv_event_x] += 1

                    label = 0
                    if prop_mask[dv_event_y, dv_event_x]:
                        label = 1

                    f_labelled_timestamp.append([dv_event_timestamp])
                    f_labelled_polarity.append([dv_event_polarity])
                    f_labelled_x.append([dv_event_x])
                    f_labelled_y.append([dv_event_y])
                    f_labelled_label.append([label])

                dv_event_timestamp, dv_event_polarity, dv_event_x, dv_event_y = get_next_dv_event(
                    f_dv_event_timestamp, f_dv_event_polarity, f_dv_event_x, f_dv_event_y,
                )

        except StopIteration:
            break

        # fill DV event image with events, then mask it
        event_image.fill(0)
        #event_image[prop_mask.astype('bool')] = grey # show prop mask?
        if distinguish_dv_event_polarity:
            event_mask_neg = event_neg > event_pos
            event_image[(event_mask_neg & ~prop_mask.astype('bool'))] = green
            event_image[(event_mask_neg & prop_mask.astype('bool'))] = blue
            event_mask_pos = event_pos > event_neg
            event_image[(event_mask_pos & ~prop_mask.astype('bool'))] = red
            event_image[(event_mask_pos & prop_mask.astype('bool'))] = yellow
        else:
            event_mask = event_neg.astype('bool') | event_pos.astype('bool')
            event_image[(event_mask & ~prop_mask.astype('bool'))] = green
            event_image[(event_mask & prop_mask.astype('bool'))] = red

        # write and show DV event image
        event_image_video.write(event_image)
        # cv2.imshow('event image', event_image)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     break


        # get next DV frame
        try:
            while dv_frame_timestamp_b < vicon_timestamp:
                dv_frame_timestamp_a, dv_frame_timestamp_b, dv_frame_image = get_next_dv_frame(
                    f_dv_frame_timestamp_a, f_dv_frame_timestamp_b, f_dv_frame_image,
                )

        except StopIteration:
            break

        # get DV frame image, then mask it
        frame_image[:, :, :] = dv_frame_image
        frame_image[prop_mask.astype('bool'), :] = blue

        # write and show DV frame image
        frame_image_video.write(frame_image)
        # cv2.imshow('frame image', frame_image)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     break


    event_image_video.release()
    frame_image_video.release()

    f_labelled.close()
    f_dv_event.close()
    f_dv_frame.close()
    f_vicon.close()
    return


if __name__ == '__main__':
    projection()
