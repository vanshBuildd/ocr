from PIL import Image
import pytesseract
import argparse
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 180, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)


prompt2 = f"""
You are an AI assistant specialized in extracting structured information from OCR-processed text. Your task is to analyze the following OCR-extracted text and create a HTML table with the following columns: Product Details, HSN Code, Rate, GST, NOS, Quantity, and Amount.

Instructions:
1. Carefully read the OCR-extracted text provided.
2. Identify information related to products, HSN Codes, Rates, GST, NOS (if applicable), quantities, and amounts.
3. Organize this information into an HTML table format.
4. If any information is missing or unclear, mark it as "N/A" in the table.
5. PLEASE DO NOT WRITE ANYTHING ELSE APART FROM THE HTML TABLE

Please provide the extracted information in the following HTML table format:

<table border="1">
    <tr>
        <th>Product Details</th>
        <th>HSN Code</th>
        <th>Rate</th>
        <th>GST</th>
        <th>NOS</th>
        <th>Quantity</th>
        <th>Amount</th>
    </tr>
    <tr>
        <td>[Product 1]</td>
        <td>[HSN Code 1]</td>
        <td>[Rate 1]</td>
        <td>[GST 1]</td>
        <td>[NOS 1]</td>
        <td>[Quantity 1]</td>
        <td>[Amount 1]</td>
    </tr>
    <!-- More rows as needed -->
</table>
    """
