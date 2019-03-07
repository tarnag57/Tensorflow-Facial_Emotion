import pandas as pd
import azure_query_helper
import time

# Labels of selfie images
names = [
    'image_name',
    'popularity',
    'is_partial',   # => only use complete images
    'is_female',    
    'is_baby',
    'is_child',
    'is_teenager',
    'is_youth',
    'is_middle_age',
    'is_senior',
    'is_white',
    'is_black',
    'is_asian',
    'is_oval_face',
    'is_round_face',
    'is_smiling',
    'is_mouth_open',
    'is_frowning',
    'is_wearing_glasses',
    'is_wearing_sunglasses',
    'is_wearing_sunglasses',
    'is_wearing_lipstick',
    'is_tongue_out',    # => new emoji symbol :P
    'is_duck_face',     # => new emoji symbol *kissing*
    'is_black_hair',
    'is_blond_hair',
    'is_brown_hair',
    'is_red_hair',
    'is_curly_hair',
    'is_straight_hair',
    'is_braid_hair',
    'is_showing_cellphone',     # => filtered
    'is_using_earphone',
    'is_using_mirror',          # => filtered
    'is_braces',
    'is_wearing_hat', 
    'is_harsh_lighting',
    'is_dim_lighting'
]

output_labels=[
    'image_name',
    'face_top',
    'face_left',
    'face_width',
    'face_height',
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happiness',
    'neutral',
    'sadness',
    'surprise',
    'is_tongue_out',
    'is_duck_face'
]

df = pd.read_csv("Selfie-dataset/selfie_dataset.txt", sep=" ", names=names)

# Filtering for not suitable images
filtered = df.loc[(df['is_partial'] < 0) & (df['is_showing_cellphone'] < 0) & (df['is_using_mirror'] < 0)]
filtered = filtered.reset_index(drop=True)
sample = filtered.loc[6000:6499]
filename = "selected_6000_6499.csv"
cycle = 0

# Storing query results
query_results = pd.DataFrame(columns=output_labels)

not_found_count = 0

# Looping over rows
try:
    for index, row in sample.iterrows():

        if cycle % 20 == 0 and cycle > 0:
            print("Finished {} queries".format(index))
            print("\n")
            print("Waiting for batch no. {}".format(index / 20))
            time.sleep(57)

        # Query image
        cycle += 1
        image_path = "Selfie-dataset/images/{}.jpg".format(row['image_name'])
        raw_json = azure_query_helper.query_picture(image_path)
        if not raw_json:
            print("Could not find face on {}".format(row['image_name']))
            not_found_count += 1
            continue

        try:
            raw_face_data = raw_json[0]

            # Parsing raw JSON
            face_rect = raw_face_data['faceRectangle']
            emotions = raw_face_data['faceAttributes']['emotion']

            entry = {
                'image_name': row['image_name'],
                'face_top': face_rect['top'],
                'face_left': face_rect['left'],
                'face_width': face_rect['width'],
                'face_height': face_rect['height'],
                'anger': emotions['anger'],
                'contempt': emotions['contempt'],
                'disgust': emotions['disgust'],
                'fear': emotions['fear'],
                'happiness': emotions['happiness'],
                'neutral': emotions['neutral'],
                'sadness': emotions['sadness'],
                'surprise': emotions['surprise'],
                'is_tongue_out': row['is_tongue_out'],
                'is_duck_face': row['is_duck_face']
            }
            query_results.loc[index] = entry

        except:
            print("An error happened while parsing {}".format(row['image_name']))
            print(raw_json)
            print("\n")

    print("\n\n")
    print("========================")
    print("Finished querying.")
    print("{} faces were not found".format(not_found_count))
    print(query_results)

    query_results.to_csv(filename, index=None, header=True)

except Exception:
    print("An error occurred, saving dataframe so-far...")
    query_results.to_csv(filename, index=None, header=True)
    print(Exception)
