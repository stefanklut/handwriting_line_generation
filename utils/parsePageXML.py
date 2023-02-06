import re
import sys
import xml.etree.ElementTree as ET 
from xml.sax.saxutils import unescape as unescape_
import json
from collections import defaultdict
import numpy as np
from pathlib import Path
#import imageio

def unescape(s):
    return unescape_(s).replace('&quot;','"').replace('&apos;',"'")


def getLineBoundaries(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    xmlns_match = re.match(r"{.*}", root.tag)
    if xmlns_match is None:
        raise ValueError
    xmlns = xmlns_match.group(0)[1:-1]
    namespace = {"xmlns": xmlns}
    pageLines=defaultdict(list)
    
    for page in root.findall('xmlns:Page', namespace):
        image = page.attrib['imageFilename']
        # image = image[image.index('/')+1:]
        allHs=0
        lines=[]
        
        # TODO Change if we do add authors
        author = image
        
        for line in page.findall(f'xmlns:TextRegion/xmlns:TextLine', namespace):
            
            text_equiv = line.find('xmlns:TextEquiv/xmlns:Unicode', namespace)
            # if text_equiv is None:
            #     raise ValueError(f"{xmlPath} is missing a text equivalent at line id {line.attrib['id']}")
            # if text_equiv.text is None:
            #     raise ValueError(f"{xmlPath} text equivalent is empty at line id {line.attrib['id']}")
            if text_equiv is None or text_equiv.text is None or text_equiv.text == "":
                continue
            trans = unescape(text_equiv.text)
            
            str_coords = line.find('xmlns:Coords', namespace)
            # Error if None
            coords = np.array([i.split(",") for i in str_coords.attrib['points'].split()]).astype(np.int32)
            # print(coords.shape)
            # print(coords)
            
            top = np.min(coords[:, 1])
            bot = np.max(coords[:, 1])
            left = np.min(coords[:, 0])
            right = np.max(coords[:, 0])
            
            lines.append(([top,bot+1,left,right+1],trans))
            allHs+=1+bot-top
        # if len(lines) == 0:
        #     raise ValueError(f"{xmlPath} no lines found")
        if len(lines) == 0:
            continue
        meanH = allHs/len(lines)
        
        xml_path = Path(xmlPath)
        
        image = str(xml_path.absolute().parents[1].joinpath(image))
        for bounds,trans in lines:
            diff = meanH-(bounds[1]-bounds[0])
            if diff>0:
                #pad out to make short words the same height on the page
                bounds[0]-=diff/2
                bounds[1]+=diff/2
            #but don't clip in tall words

            #add a little extra padding horizontally
            bounds[2]-= meanH/4
            bounds[3]+= meanH/4
            bounds = [round(v) for v in bounds]
            #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            pageLines[author].append((image,bounds,trans))
    if len(pageLines) == 0:
        print(f"WARNING: No lines found in xml ({xmlPath})")
    return pageLines


if __name__ == "__main__":
    xml_path = "/home/stefan/Documents/datasets/ijsberg/images/page/NL-0400410000_26_004011_000464.xml"
    getLineBoundaries(xml_path)
    