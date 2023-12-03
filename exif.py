# adapted from https://thepythoncode.com/article/extracting-image-metadata-in-python

from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffTags import TAGS_V2
import exifread
import numpy as np

def get_gps_data(imagename):
    """Get GPS data from image.

    Args:
        imagename (str): 

    Returns:
        Tuple(float, float, float): Latitude (decimals), longitude (decimals), altitude (meters above sea level).
    """
    # Open image file for reading (binary mode)
    f = open(imagename, 'rb')
    tags = exifread.process_file(f)
    lat, long, alt = None, None, None

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

def gps_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """Get distance between two GPS coordinates in kilometers.

    Args:
        lat1 (_type_): in degress
        lon1 (_type_): in degress
        alt1 (_type_): in meters above sea level
        lat2 (_type_): in degress
        lon2 (_type_): in degress
        alt2 (_type_): in meters above sea level

    Returns:
        float: distance in kilometers
    """
    # d = 2R × sin⁻¹(√[sin²((θ₂ - θ₁)/2) + cosθ₁ × cosθ₂ × sin²((φ₂ - φ₁)/2)])
    # where:

    # θ₁, φ₁ – First point latitude and longitude coordinates;
    # θ₂, φ₂ – Second point latitude and longitude coordinates;
    # R – Earth's radius (R = 6371 km); and
    # d – Distance between them along Earth's surface.
    # convert to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    horizontal = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))

    # horizontal = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))*6371
    vertical = abs(alt1 - alt2)/1000 # convert to km
    hypotenuse = np.sqrt(horizontal**2 + vertical**2)
    return hypotenuse

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
    lat1, lon1, alt1 = get_gps_data('data/cathedral/IMG_6303.HEIC')
    lat2, lon2, alt2 = get_gps_data('data/cathedral/IMG_6304.HEIC')
    print(lat1, lon1, alt1)
    print(lat2, lon2, alt2)
    print(gps_distance(lat1, lon1, alt1, lat2, lon2, alt2))
