import os
import tempfile

from cyclonedds.core import Policy, Qos
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


def telemetry_qos():
    return Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1), Policy.Durability.Volatile)


def configure_interface(iface):
    """Pin CycloneDDS to one network interface; returns the config path or None."""
    if not iface or 'CYCLONEDDS_URI' in os.environ:
        return None

    config_content = f'''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain>
        <General>
            <Interfaces>
                <NetworkInterface name="{iface}"/>
            </Interfaces>
        </General>
    </Domain>
</CycloneDDS>'''

    fd, config_file = tempfile.mkstemp(suffix='.xml', prefix='cyclonedds_', text=True)
    os.close(fd)
    with open(config_file, 'w') as f:
        f.write(config_content)
    os.environ['CYCLONEDDS_URI'] = f'file://{config_file}'
    return config_file


def writer(participant, topic_name, msg_type):
    return DataWriter(participant, Topic(participant, topic_name, msg_type), qos=telemetry_qos())


def reader(participant, topic_name, msg_type):
    return DataReader(participant, Topic(participant, topic_name, msg_type), qos=telemetry_qos())
