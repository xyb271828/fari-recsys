"""Download original public benchmark files without committing user data."""
import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


def download(url, destination, algorithm, expected):
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=120) as source, open(partial, 'wb') as output:
            shutil.copyfileobj(source, output)
        partial.replace(destination)
    digest = hashlib.new(algorithm)
    with open(destination, 'rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    download('https://files.grouplens.org/datasets/movielens/ml-1m.zip', args.output/'ml-1m.zip',
             'sha256', 'a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20')
    archive = args.output/'KuaiRand-Pure.tar.gz'
    download('https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz', archive,
             'md5', '0820331067a3784d9691136f772b35a7')
    with tarfile.open(archive) as source:
        for name in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv',
                     'log_random_4_22_to_5_08_pure.csv'):
            member = source.getmember('KuaiRand-Pure/data/' + name)
            if not member.isfile():
                raise ValueError('Expected a regular CSV file')
            destination = args.output/'KuaiRand-Pure/data'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, open(destination, 'wb') as output:
                shutil.copyfileobj(stream, output)
    print(f'Verified benchmark files available in {args.output}')


if __name__ == '__main__':
    main()
