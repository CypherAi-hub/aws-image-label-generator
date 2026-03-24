import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

def detect_labels(photo, bucket):
    client = boto3.client('rekognition', region_name='us-east-2')

    response = client.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MaxLabels=10
    )

    print(f"\nDetected labels for {photo}:\n")

    for label in response['Labels']:
        print(f"{label['Name']} : {label['Confidence']:.2f}%")

    # Load image
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, photo)
    img_data = obj.get()['Body'].read()
    img = Image.open(BytesIO(img_data))

    plt.imshow(img)
    ax = plt.gca()

    for label in response['Labels']:
        for instance in label.get('Instances', []):
            bbox = instance['BoundingBox']

            left = bbox['Left'] * img.width
            top = bbox['Top'] * img.height
            width = bbox['Width'] * img.width
            height = bbox['Height'] * img.height

            rect = patches.Rectangle(
                (left, top), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            label_text = f"{label['Name']} ({label['Confidence']:.2f}%)"

            plt.text(
                left, top - 5,
                label_text,
                color='red',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )

    plt.axis('off')
    plt.savefig(f"output_{photo}.png", bbox_inches='tight')
    plt.show()

    return len(response['Labels'])


def main():
    bucket = 'amzn-image-rekognition-labeler'
    photos = ['apple.png']

    for photo in photos:
        detect_labels(photo, bucket)


if __name__ == "__main__":
    main()