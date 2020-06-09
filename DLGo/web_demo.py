import argparse

import h5py

from dlgo import agent
from dlgo import httpfrontend
from dlgo import zero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    args = parser.parse_args()
    
    zero_agent = {'zero': zero.load_zero_agent(h5py.File("bots/zeroagent_v1.hdf5", 'r'))}

    web_app = httpfrontend.get_web_app(zero_agent)
    web_app.run(host=args.bind_address, port=args.port, threaded=False)


if __name__ == '__main__':
    main()