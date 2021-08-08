<h2>Audio Spectral Enhancement: Leveraging Autoencoders for Low Latency Reconstruction of Long, Lossy Audio Sequences</h2>

This package contains the code proposed in <a href="">Audio Spectral Enhancement: Leveraging Autoencoders for Low Latency Reconstruction of Long, Lossy Audio Sequences</a> (Darshan Deshpande and Harshavardhan Abichandani, 2021).


<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
![version-shield]
![release-shield]
![python-shield]
![code-style]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citations">Citations</a></li>
  </ol>
</details>

## Prerequisites

Prerequisites can be installed separately through the `requirements.txt` file as below

```sh
pip install -r requirements.txt
```


<!-- USAGE EXAMPLES -->
## Usage

The following code can be used to load and process the GTZAN dataset as is done in the paper
```sh
python create_dataset.py
```
Once done, you will have two directories in your current path. To start training the model, execute the `train.py` file. The training parameters can be tweaked as per your requirement. See ` python train.py --help` for the training options.

```sh
python train.py --audio-low-path 'FFMPEGConvertedCut32' --audio-high-path 'CutSounds128' --epochs=200
```

For inference, you can use
```sh
!python inference.py --checkpoint-path="/path/to/checkpoint.h5" --mp3-audio-file="/path/to/mp3file"
```

<!-- CONTRIBUTING -->
## Contributing

Any and all contributions are welcome. Please raise an issue if you face any issue with the training or inference code. 
<br>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Feel free to reach out for any issues or requests related to this implementation

Darshan Deshpande - [Email](https://mail.google.com/mail/u/0/?view=cm&fs=1&to=darshan.g.deshpande@gmail.com&tf=1) | [LinkedIn](https://www.linkedin.com/in/darshan-deshpande/) <br>
Harshavardhan Abichandani - [Email](https://mail.google.com/mail/u/0/?view=cm&fs=1&to=harshavardhan.abichandani@gmail.com&tf=1) | [LinkedIn](https://www.linkedin.com/in/harsh-abhi/)




<!-- ACKNOWLEDGEMENTS -->
## Citations
GTZAN Dataset
```citation
@misc{tzanetakis_essl_cook_2001,
        author    = "Tzanetakis, George and Essl, Georg and Cook, Perry",
        title     = "Automatic Musical Genre Classification Of Audio Signals",
        url       = "http://ismir2001.ismir.net/pdf/tzanetakis.pdf",
        publisher = "The International Society for Music Information Retrieval",
        year      = "2001"
    }
}
```





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/CONTRIBUTORS-1-orange?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[license-shield]: https://img.shields.io/badge/LICENSE-MIT-brightgreen?style=for-the-badge
[license-url]: https://github.com/DarshanDeshpande/tf-madgrad/blob/master/LICENSE.txt
[version-shield]: https://img.shields.io/badge/VERSION-1.0.0-orange?style=for-the-badge
[python-shield]: https://img.shields.io/badge/PYTHON-3.6%7C3.7%7C3.8-blue?style=for-the-badge
[release-shield]: https://img.shields.io/badge/Build-Stable-yellow?style=for-the-badge
[code-style]: https://img.shields.io/badge/Code_Style-Black-black?style=for-the-badge
