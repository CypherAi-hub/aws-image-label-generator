import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

REGION = "us-east-2"
BUCKET = "amzn-image-rekognition-labeler"


def list_s3_images(bucket):
    s3 = boto3.client("s3", region_name=REGION)
    response = s3.list_objects_v2(Bucket=bucket)

    image_files = []
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".avif")

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.lower().endswith(valid_exts):
            image_files.append(key)

    return sorted(image_files)


def get_personalized_insights(labels):
    label_names = {label["Name"].lower() for label in labels}
    insights = []

    if "person" in label_names:
        insights.append("Fitness insight: person detected — this could be used for workout or movement analysis.")
    if "shoe" in label_names or "footwear" in label_names:
        insights.append("Lifestyle insight: shoe detected — possible running, training, or athletic gear.")
    if "beach" in label_names or "ocean" in label_names or "sea" in label_names:
        insights.append("Scene insight: outdoor environment detected — travel or lifestyle content.")
    if "burger" in label_names or "food" in label_names:
        insights.append("Nutrition insight: food item detected — could be expanded into meal recognition.")
    if "dog" in label_names or "animal" in label_names:
        insights.append("General insight: animal detected — good for testing non-fitness image accuracy.")
    if "computer" in label_names or "laptop" in label_names:
        insights.append("Tech insight: computer detected — useful for office, productivity, or tech scene analysis.")

    if not insights:
        insights.append("No custom lifestyle insight matched yet — good opportunity to expand your categories.")

    return insights


def detect_labels(photo, bucket):
    rekognition = boto3.client("rekognition", region_name=REGION)

    response = rekognition.detect_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": photo}},
        MaxLabels=10
    )

    print(f"\n📸 Image selected: {photo}")
    print("\n🔍 Detected Labels:")
    for label in response["Labels"]:
        print(f" - {label['Name']} ({label['Confidence']:.2f}%)")

    print("\n🧠 Personalized Insights:")
    for insight in get_personalized_insights(response["Labels"]):
        print(f" - {insight}")

    # Load image from S3
    s3 = boto3.resource("s3", region_name=REGION)
    obj = s3.Object(bucket, photo)
    img_data = obj.get()["Body"].read()
    img = Image.open(BytesIO(img_data))

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    for label in response["Labels"]:
        for instance in label.get("Instances", []):
            bbox = instance["BoundingBox"]

            left = bbox["Left"] * img.width
            top = bbox["Top"] * img.height
            width = bbox["Width"] * img.width
            height = bbox["Height"] * img.height

            rect = patches.Rectangle(
                (left, top),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none"
            )
            ax.add_patch(rect)

            label_text = f"{label['Name']} ({label['Confidence']:.2f}%)"
            plt.text(
                left,
                max(0, top - 5),
                label_text,
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7)
            )

    plt.axis("off")

    safe_name = photo.replace("/", "_").replace(" ", "_")
    output_name = f"output_{safe_name}.png"
    plt.savefig(output_name, bbox_inches="tight")
    print(f"\n💾 Saved output image as: {output_name}")

    plt.show()

    return len(response["Labels"])


def main():
    print("\n🔥 Welcome to FitVision AI")
    print("AI-powered image analysis with AWS Rekognition + personalized insights\n")

    images = list_s3_images(BUCKET)

    if not images:
        print("No images found in the S3 bucket.")
        return

    print("Available images in your S3 bucket:")
    for i, image_name in enumerate(images, start=1):
        print(f"{i}. {image_name}")

    try:
        choice = int(input("\nEnter the number of the image you want to analyze: "))
        if choice < 1 or choice > len(images):
            print("Invalid choice.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return

    selected_image = images[choice - 1]
    label_count = detect_labels(selected_image, BUCKET)
    print(f"\n✅ Total labels detected: {label_count}")


if __name__ == "__main__":
    main()