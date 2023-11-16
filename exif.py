# adapted from https://thepythoncode.com/article/extracting-image-metadata-in-python

from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffTags import TAGS_V2
import exifread



def get_exposure_time(imagename):
    # Open image file for reading (binary mode)
    f = open(imagename, 'rb')
    tags = exifread.process_file(f)
    val = tags['EXIF ExposureTime']
    exposure_time = val.values[0].num / val.values[0].den

    # # Print the tag/ value pairs
    # for tag in tags.keys():
    #     if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
    #         print("Key: %s, value %s" % (tag, tags[tag]))

    return exposure_time


def get_exif_data(imagename):
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

