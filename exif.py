# adapted from https://thepythoncode.com/article/extracting-image-metadata-in-python

from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffTags import TAGS_V2
import exifread
import numpy as np
import math

def get_gps_data(imagename):
    """Get GPS data from image.

    Args:
        imagename (str): 

    Returns:
        Tuple(float, float, float): Latitude (degrees), longitude (degrees), altitude (meters above sea level).
    """
    lat, long, alt = None, None, None
    # Open image file for reading (binary mode)
    with open(imagename, 'rb') as f:
        tags = exifread.process_file(f)

    # Latitude
    val = tags.get('GPS GPSLatitude', None)
    if val is not None:
        degree, minute, second = val.values
        degree, minute, second_num, second_den = np.array([degree, minute, second.num, second.den], dtype=np.float64)
        lat = degree + minute / 60 + second_num / second_den / 3600
        if tags['GPS GPSLatitudeRef'].values == 'S':
            lat = -lat

    # Longitude
    val = tags.get('GPS GPSLongitude', None)
    if val is not None:
        degree, minute, second = val.values
        degree, minute, second_num, second_den = np.array([degree, minute, second.num, second.den], dtype=np.float64)
        long = degree + minute / 60 + second_num / second_den / 3600
        if tags['GPS GPSLongitudeRef'].values == 'W':
            long = -long
    
    # Altitude
    val = tags.get('GPS GPSAltitude', None)
    if val is not None:
        alt = val.values[0].num / val.values[0].den
    else:
        alt = 0
        
    return lat, long, alt

def get_direction_data(imagename):
    """Get direction data from image in degrees.
    N is at 0 degrees, E is at 90 degrees, S is at 180 degrees, and W is at 270 degrees."""
    direction = None
    # Open image file for reading (binary mode)
    with open(imagename, 'rb') as f:
        tags = exifread.process_file(f)

    val = tags.get('GPS GPSImgDirection', None)
    if val is not None:
        direction = float(val.values[0])
    return direction

def gps_distance(
    lat1, lon1, alt1, 
    lat2, lon2, alt2
    ):
    """Get distance between two GPS coordinates in meters.

    Args:
        lat1 (float): in degress
        lon1 (float): in degress
        alt1 (float): in meters above sea level
        lat2 (float): in degress
        lon2 (float): in degress
        alt2 (float): in meters above sea level

    Returns:
        float: distance in meters
    """
    # d = 2R × sin⁻¹(√[sin²((θ₂ - θ₁)/2) + cosθ₁ × cosθ₂ × sin²((φ₂ - φ₁)/2)])
    # where:

    # θ₁, φ₁ – First point latitude and longitude coordinates;
    # θ₂, φ₂ – Second point latitude and longitude coordinates;
    # R – Earth's radius (R = 6371 km); and
    # d – Distance between them along Earth's surface.
    # convert to radians
    # horizontal = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))

    # # horizontal = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))*6371
    # vertical = abs(alt1 - alt2) # in meters
    # hypotenuse = np.sqrt(horizontal**2 + vertical**2)
    # return hypotenuse
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # assume flat earth
    # Calculate the differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the distance using the Haversine formula
    R = 6371000 # radius of earth in meters
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def get_relative_rotation(direction1, direction2):
    """relative viewing direction"""
    relative_rotation = direction1 - direction2
    # Ensure the relative_rotation is within the range [-180, 180]
    # where positive values indicate clockwise rotation, and negative values indicate counterclockwise rotation
    relative_rotation = (relative_rotation + 180) % 360 - 180
    relative_rotation_rad = math.radians(relative_rotation)
    # Calculate the elements of the rotation matrix
    cos_theta = math.cos(relative_rotation_rad)
    sin_theta = math.sin(relative_rotation_rad)

    # Create the 3D rotation matrix, assuming Y-axis is fixed
    # This assumes that the ground is the x-z plane.
    rotation_matrix = [
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, -cos_theta],
    ]
    rotation_matrix = np.array(rotation_matrix)
    # return relative rotation (deg) and 3d rotation matrix
    return relative_rotation, rotation_matrix


def get_relative_translation(
    lat1, lon1, alt1, 
    lat2, lon2, alt2,
):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    R = 6371000 # radius of earth in meters
    # convert gps coordinates to cartesian coordinates
    # x, y, z are the cartesian coords of the relative translation vector
    x = (lon2 - lon1) * np.cos(lat1) * R
    y = alt2 - alt1
    z = (lat2 - lat1) * R

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return np.array((x, y, z)).reshape((-1,1)), magnitude


def get_exposure_time(imagename):
    # Open image file for reading (binary mode)
    f = open(imagename, 'rb')
    tags = exifread.process_file(f)
    val = tags['EXIF ExposureTime']
    exposure_time = val.values[0].num / val.values[0].den

    return exposure_time

def get_exif_data1(imagename):
    f = open(imagename, 'rb')
    tags = exifread.process_file(f)
    # Print the tag/ value pairs
    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            print("Key: %s, value %s" % (tag, tags[tag]))

def get_exif_data2(imagename):
    # read the image data using PIL
    image = Image.open(imagename)

    print(image.info.keys())

    # extract other basic metadata
    info_dict = {
        "Filename": image.filename,
        "Image Size": image.size,
        "Image Height": image.height,
        "Image Width": image.width,
        "Image Format": image.format,
        "Image Mode": image.mode,
        "Image is Animated": getattr(image, "is_animated", False),
        "Frames in Image": getattr(image, "n_frames", 1)
    }

    # extract EXIF data
    exifdata = image.getexif()

    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            try:
                data = data.decode()
            except UnicodeDecodeError as e:
                data = str(data)

        info_dict[tag] = data
        # print(f"{tag:25}: {data}")

    # for k, v in image.tag_v2.items():
    #     print(k, v, TAGS_V2.get(k, k))

    # from PIL.TiffTags import TAGS

    # print(TAGS[33434])


    f = open(imagename, 'rb')
    tags = exifread.process_file(f)
    val = tags.get('EXIF ExposureTime', None)
    if val is not None:
        exposure_time = val.values[0].num / val.values[0].den
    info_dict['EXIF ExposureTime_decimal'] = exposure_time
    info_dict.update(tags)

    return info_dict


if __name__ == '__main__':
    im1_path, im2_path = ('data/cathedral/IMG_6343.jpeg', 'data/cathedral/IMG_6346.jpeg')
    lat1, lon1, alt1 = get_gps_data(im1_path)
    lat2, lon2, alt2 = get_gps_data(im2_path)
    dir1 = get_direction_data(im1_path)
    dir2 = get_direction_data(im2_path)
    print(f"{(lat1, lon1, alt1, dir1)=}")
    print(f"{(lat2, lon2, alt2, dir2)=}")
    print(f"{gps_distance(lat1, lon1, alt1, lat2, lon2, alt2)=}")
    print(f"{get_relative_rotation(dir1, dir2)=}")
    print(f"{get_relative_translation(lat1, lon1, alt1, lat2, lon2, alt2)=}")
