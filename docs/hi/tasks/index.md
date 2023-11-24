---
comments: true
description: जानें YOLOv8 जो कि विभिन्न कंप्यूटर विजन कार्यों जैसे डिटेक्शन, सेग्मेंटेशन, क्लासिफिकेशन और पोज़ एस्टिमेशन को कर सकता है| अपनें AI प्रोजेक्ट्स म इन टास्क का उपयोग के बारें म मर्यादित हो जाएं
keywords: Ultralytics, YOLOv8, डिटेक्शन, सेग्मेंटेशन, क्लासिफिकेशन, पोज़ एस्टिमेशन, AI Framework, कंप्यूटर विजन कार्य
---

# Ultralytics YOLOv8 तास्क

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO Supported टास्क्स">

YOLOv8 एक AI फ्रेमवर्क है जो मल्टीपल कंप्यूटर विजन **तास्क्स** को सपोर्ट करता है। इस फ्रेमवर्क का उपयोग [डिटेक्शन](detect.md), [सेग्मेंटेशन](segment.md), [क्लासिफिकेशन](classify.md), और [पोज़](pose.md) एस्टिमेशन को करने के लिए किया जा सकता हैं। हर टास्क का एक अलग उद्देश्य और यूज केस होता हैं।

!!! Note "नोट"

    🚧 हमारा मल्टी-भाषा डॉक्युमेंटेशन वर्तमान में निर्माणाधीन हैं, और हम उसे सुधारने के लिए मेहनत कर रहें हैं। आपकी सहानुभूति के लिए धन्यवाद! 🙏

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> जांचें Ultralytics YOLO टास्क्स: वस्तु डिटेक्शन, सेग्मेंटेशन, ट्रैकिंग और पोज़ एस्टिमेशन।
</p>

## [डिटेक्शन](detect.md)

डिटेक्शन YOLOv8 द्वारा सपोर्ट किया जाने वाला प्राथमिक टास्क हैं। इसका मतलब होता हैं कि एक छवि या वीडियो फ्रेम में वस्तुओं को खोजें और उनके चारों ओर ग्रेडीयेशन बॉक्स बनाएँ। पायी गयी वस्तुओं को उनके फीचर्स के आधार पर विभिन्न श्रेणियों में वर्गीकृत किया जाता हैं। YOLOv8 एक ही छवि या वीडियो फ्रेम में कई वस्तुएं पहचान सकती हैं और उसे उच्च सटीकता और गति से कर सकती हैं।

[डिटेक्शन उदाहरण](detect.md){ .md-button .md-button--primary}

## [सेग्मेंटेशन](segment.md)

सेग्मेंटेशन एक टास्क हैं जिसमे एक छवि को उसकी सामग्री के आधार पर विभिन्न क्षेत्रों में विभाजित किया जाता हैं। प्रत्येक क्षेत्र को उसकी सामग्री के आधार पर एक लेबल दिया जाता हैं। यह टास्क छवि सेग्मेंटेशन और मेडिकल इमेजिंग जैसे एप्लिकेशन्स में उपयोगी होती हैं। YOLOv8 सेग्मेंटेशन करने के लिए U-Net आर्किटेक्चर का इस्तेमाल करता हैं।

[सेग्मेंटेशन उदाहरण](segment.md){ .md-button .md-button--primary}

## [क्लासिफिकेशन](classify.md)

क्लासिफिकेशन एक टास्क हैं जिसमे एक छवि को विभिन्न श्रेणियों में वर्गीकृत किया जाता हैं। YOLOv8 का उपयोग छवियों को उनकी सामग्री के आधार पर क्लासिफाई करने के लिए किया जा सकता हैं। यह क्लासिफिकेशन करने के लिए EfficientNet आर्किटेक्चर का उपयोग करता हैं।

[क्लासिफिकेशन उदाहरण](classify.md){ .md-button .md-button--primary}

## [पोज़](pose.md)

पोज़/कीपॉइंट डिटेक्शन एक टास्क हैं जिसमे एक छवि या वीडियो फ्रेम में विशेष बिंदुओं को खोजें। इन बिंदुओं को कीपॉइंट कहा जाता हैं और इनका उपयोग गति या पोज़ एस्टिमेशन करने के लिए किया जाता हैं। YOLOv8 एक छवि या वीडियो फ्रेम में उच्च सटीकता और गति से कीपॉइंट डिटेक्ट कर सकता हैं।

[पोज़ उदाहरण](pose.md){ .md-button .md-button--primary}

## निष्कर्ष

YOLOv8 डिटेक्शन, सेग्मेंटेशन, क्लासिफिकेशन और कीपॉइंट डिटेक्शन जैसे मल्टीपल टास्क्स को सपोर्ट करता हैं। हर एक टास्क का अलग उद्देश्य और यूज केस होता हैं। इन टास्क्स के बीच अंतर को समझकर, आप अपने कंप्यूटर विजन एप्लिकेशन के लिए उचित टास्क का चुनाव कर सकते हैं।