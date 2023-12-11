from bs4 import BeautifulSoup
import requests
import os

URL = "http://www.eng.usf.edu/cvprg/Mammography/DDSM/thumbnails/benigns/benign_11/overview.html"
r = requests.get(URL)

soup = BeautifulSoup(r.content, 'html5lib')

imgs = []

# Iterate through all the links on the webpage
for link in soup.findAll('a'):
    # If the link ends with .html, it's a webpage
    if link['href'].endswith('.html') and not 'figment' in link['href']:
        # Get this page
        link_path =f"{URL.rsplit('/', 1)[0]}/{link['href']}"
        r = requests.get(link_path)

        # Parse the page
        soup2 = BeautifulSoup(r.content, 'html5lib')

        # Get the h2 tag, which contains the image name
        h2 = soup2.find('h2').text

        img_arr = []
        img_arr.append(h2)

        # print the urls of all images found on a webpage
        for image in soup2.findAll('img')[:2]:
            # Combine the url of the page with the relative path of the image
            # Remove the *.html from the url, split after the last /
            # and add the relative path of the image

            img_arr.append(f"{link_path.rsplit('/', 1)[0]}/{image['src']}")

        imgs.append(img_arr)
print(imgs)

for volume, img1, img2 in imgs:
    # print(volume, img1, img2)
    # pass
    # Create a folder for each volume
    # Download the images to the folder
    # Use the volume name as the folder name

    # Create a folder for each volume
    os.mkdir(f"images/{volume}")
    # Download the images to the folder
    # Use the volume name as the folder name
    with open(f"images/{volume}/{img1.rsplit('/', 1)[1]}", 'wb') as f:
        f.write(requests.get(img1).content)
    
    print(f"images/{volume}/{img2.rsplit('/', 1)[1]} downloaded")

    with open(f"images/{volume}/{img2.rsplit('/', 1)[1]}", 'wb') as f:
        f.write(requests.get(img2).content)

    print(f"images/{volume}/{img2.rsplit('/', 1)[1]} downloaded")
