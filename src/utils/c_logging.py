import json
import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
import os
import socket
import sys


def log_json(logger, log_level, type_name, json):
    if logger.isEnabledFor(log_level):
        logger.log(log_level, 'JSON - %s - %s', type_name, json)


def log_item(logger, log_level, type_name, item):
    if logger.isEnabledFor(log_level):
        log_json(logger, log_level, type_name, item if isinstance(item, dict) else json.dumps(item.__dict__))


def getLogger(name):
    logger = logging.getLogger(name)
    logger.log_item = lambda log_level, type_name, item: log_item(logger, log_level, type_name, item)
    logger.log_json = lambda log_level, type_name, item: log_json(logger, log_level, type_name, item)
    return logger


def config(project, task, experiment_id, experiment_dir, log_level=logging.INFO):
    log = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    file_handler = RotatingFileHandler(os.path.join(experiment_dir, '{}-{}-{}.log'.format(project, task, experiment_id)))
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.setLevel(log_level)
    log.info('Logging %s %s %s', project, task, experiment_id)


def config_docker_local(perimeter, prog_name, log_level=logging.INFO):
    log = logging.getLogger()
    platform = os.environ.get("PLATFORM", "local")
    container_id = os.environ.get("CONTAINER_ID", "0")
    container_ip_address = os.environ.get("CONTAINER_IP_ADDRESS", "127.0.0.1")
    formatter = get_logging_formatter(container_ip_address, perimeter, platform, prog_name)
    add_file_handler(container_id, container_ip_address, formatter, log, log_level, perimeter, platform, prog_name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.setLevel(log_level)


def add_file_handler(container_id, container_ip_address, formatter, log, log_level, perimeter, platform, prog_name):
    log_dir = "/var/log/docker/{}/{}/{}/{}".format(perimeter, platform, prog_name, container_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = TimedRotatingFileHandler(os.path.join(
        log_dir, 
        "{}_local0_{}.log".format(container_ip_address, log_level)), 
        'midnight', 
        1)
    file_handler.suffix = "%Y-%m-%d" # or anything else that strftime will allow
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)


def get_logging_formatter(hostname, perimeter, platform, prog_name):
    formatter = logging.Formatter(
        perimeter + "/" + platform + "/" + prog_name + "/" + hostname + \
            ": %(asctime)s.%(msecs)03d [" + prog_name + ",,,] [%(thread)s] %(levelname)s a.Python - %(message)s",
            '%Y-%m-%d %H:%M:%S')
    return formatter


def config_docker_syslog(perimeter, prog_name, log_level=logging.INFO, syslog_host='172.17.0.1', syslog_port=514, log_file=False):
    platform = os.environ.get("PLATFORM", "local")
    logger = logging.getLogger()
    handler = logging.handlers.SysLogHandler(address=(syslog_host, syslog_port),
    facility=logging.handlers.SysLogHandler.LOG_LOCAL0)
    formatter = get_logging_formatter(socket.gethostname(), perimeter, platform, prog_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        container_id = os.environ.get("CONTAINER_ID", "0")
        add_file_handler(container_id, socket.gethostname(), formatter, logger, log_level, perimeter, platform, prog_name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)
