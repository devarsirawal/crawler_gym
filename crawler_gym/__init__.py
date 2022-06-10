from gym.envs.registration import register
register(
    id="Crawler-v0",
    entry_point="crawler_gym.envs:CrawlerEnv"
)