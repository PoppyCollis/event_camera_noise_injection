
import numpy as np
import tables
import stl

def create_labelled_data_file(f_name='./data/labelled.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))
    f_label = f.create_earray(f.root, 'label', tables.atom.UInt8Atom(), (0,))
    f.close()


def projection():


    test_name = 'noise'

    test_number = 1

    #record_seconds = 3
    record_seconds = 10
    #record_seconds = 100
    record_time = record_seconds * 1000000

    #vicon_usec_offset = 183000
    #vicon_usec_offset = 157000
    #vicon_usec_offset = 69000000
    vicon_usec_offset = 3030000
    #vicon_usec_offset = 0

    distinguish_dv_event_polarity = False



    dv_camera_matrix = np.load('./calibration/camera_matrix.npy')
    dv_distortion_coefficients = np.load('./calibration/camera_distortion_coefficients.npy')

    f_labelled_name = f'./data/labelled_{test_name}_{test_number:04}.h5'
    f_dv_event_name = f'./data/dv_event_{test_name}_{test_number:04}.h5'
    
    f_dv_frame_name = f'./data/dv_frame_{test_name}_{test_number:04}.h5'
    f_vicon_name = f'./data/vicon_{test_name}_{test_number:04}.h5'
    f_noise_event_name = f'./data/noise_event_{test_name}_{test_number:04}.h5'

    f_event_image_video = f'./data/event_image_video_{test_name}_{test_number:04}.avi'
    f_frame_image_video = f'./data/frame_image_video_{test_name}_{test_number:04}.avi'




 #################### noise injection #############################

    def get_next_dv_event(f_timestamp, f_polarity, f_x, f_y, usec_offset=0):
        timestamp = np.uint64(next(f_timestamp) + usec_offset)
        polarity = next(f_polarity)
        x = next(f_x)
        y = next(f_y)

        return timestamp, polarity, x, y

    def get_next_noise_event(f_timestamp, f_polarity, f_x, f_y, usec_offset=0):
        timestamp = np.uint64(next(f_timestamp) + usec_offset)
        polarity = next(f_polarity)
        x = next(f_x)
        y = next(f_y)

        return timestamp, polarity, x, y

    def create_labelled_data_file(f_name='./data/labelled.h5'):
        f = tables.open_file(f_name, mode='w')
        f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
        f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
        f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
        f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))
        f_label = f.create_earray(f.root, 'label', tables.atom.UInt8Atom(), (0,))
        f.close()


    #open dv events and noise events, write a new file with noise events plus dv events 

    dv_event = tables.open_file(f_dv_event_name, mode='r')
    dv_event_timestamp = dv_event.root.timestamp.iterrows()
    dv_event_polarity = dv_event.root.polarity.iterrows()
    dv_event_x = dv_event.root.x.iterrows()
    dv_event_y = dv_event.root.y.iterrows()

    noise_event = tables.open_file('./data/noise_event.h5', mode='r')
    noise_event_timestamp = noise_event.root.timestamp.iterrows()
    noise_event_polarity = noise_event.root.polarity.iterrows()
    noise_event_x = noise_event.root.x.iterrows()
    noise_event_y = noise_event.root.y.iterrows()

    create_labelled_data_file(f_name='./data/combo_event.h5')
    
    f = tables.open_file('./data/combo_event.h5', mode='a')
    f_timestamp = f.root.timestamp
    f_x = f.root.x
    f_y = f.root.y
    f_polarity = f.root.polarity

    #n_offset = 1626270537204981
    n_offset = 1626268040902733
    #n_offset = 0
    


    print('noise:')
    n_timestamp = (next(noise_event_timestamp)+ n_offset)
    print(n_timestamp)
    n_timestamp = (next(noise_event_timestamp)+ n_offset)
    print(n_timestamp)

    print('dv:')
    dv_timestamp = next(dv_event_timestamp)
    print(dv_timestamp)
    dv_timestamp = next(dv_event_timestamp)
    print(dv_timestamp)

    
    n_polarity = next(noise_event_polarity)
    #print(n_polarity)

    dv_polarity = next(dv_event_polarity)
    #print(dv_polarity)

    n_x = next(noise_event_x)
    #print(n_x)

    dv_x = next(dv_event_x)
    #print(dv_x)

    n_y = next(noise_event_y)
    #print(n_y)

    dv_y = next(dv_event_y)
    #print(dv_y)


    #xit(0)


    
    while True:
        try:

            if n_timestamp <= dv_timestamp:
                if n_timestamp == dv_timestamp and n_x == dv_x and n_y == dv_y:
                    print('same' * 10)
                    f_timestamp.append([dv_timestamp])
                    f_polarity.append([dv_polarity])
                    f_x.append([dv_x])
                    f_y.append([dv_y])
                    dv_timestamp = next(dv_event_timestamp)
                    dv_polarity = next(dv_event_polarity)
                    dv_x = next(dv_event_x)
                    dv_y = next(dv_event_y)

                else:
                    f_timestamp.append([n_timestamp])
                    f_polarity.append([n_polarity])
                    f_x.append([n_x])
                    f_y.append([n_y])
                    #print('noise event appended'*4)
                    #print(n_timestamp, 'x: ', n_x, 'y: ', n_y, 'p: ', n_polarity)
                    n_timestamp = (next(noise_event_timestamp) + n_offset)
                    n_polarity = next(noise_event_polarity)
                    n_x = next(noise_event_x)
                    n_y = next(noise_event_y)
              
            else:
                f_timestamp.append([dv_timestamp])
                print('dv_event_appended')
                print(dv_timestamp)
                f_polarity.append([dv_polarity])
                f_x.append([dv_x])
                f_y.append([dv_y])
                dv_timestamp = next(dv_event_timestamp)
                dv_polarity = next(dv_event_polarity)
                dv_x = next(dv_event_x)
                dv_y = next(dv_event_y)
               
        except StopIteration:
            break
                      
    f.close()
    dv_event.close()
    noise_event.close()

    return


if __name__ == '__main__':
    projection()
