# mutant

<p align='center'>
    <img src='assets/logo.png'/>
</p>

Experiments with the latest releases of tensorflow and hugging face libraries

## Running

To run the currently configured experiment:

```sh
python -m mutant run 'Two men walk into a bar. One of them says "I want H2O", the second man says "I want H2O too". The second man dies. Why?'
```

For example, when you run `fill-mask` task after executing the following command:

```sh
python -m mutant run 'Why dark is spelled with k but not with c? Because you cant <mask> in the dark'
```

You will be in response something like this:

```sh
Why dark is spelled with k but not with c? Because you cant sleep in the dark
Why dark is spelled with k but not with c? Because you cant walk in the dark
Why dark is spelled with k but not with c? Because you cant see in the dark
Why dark is spelled with k but not with c? Because you cant hide in the dark
Why dark is spelled with k but not with c? Because you cant glow in the dark
```

## Installation

To create `conda` environment and install dependencies:

```sh
conda env create -f environment.yml
```
