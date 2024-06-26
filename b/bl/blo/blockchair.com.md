> *The following text is extracted and transformed from the blockchair.com privacy policy that was archived on 2019-10-14. Please check the [original snapshot on the Wayback Machine](https://web.archive.org/web/20191014051621id_/https%3A//blockchair.com/privacy) for the most accurate reproduction.*

# Privacy policy — Blockchair

**TL;DR: Blockchair does not collect personal data or share it with third parties. We don't track you.**
    
    
    server {
      server_name blockchair.com;
      access_log off;

**One of the key advantages of cryptocurrencies is that they enable (pseudo)anonymous transactions.** In most cases the user’s address and transaction details are made public and cannot be deleted, but their personal identity remains unknown if no link exists between the user and their blockchain data.

Privacy is at risk when you share any information with third parties. Cryptocurrency exchanges with KYC policies, online retailers that require delivery addresses and web wallets associated with phone numbers all require you to share information.

What’s more, most web servers maintain default logs of your IP address and User Agent (browser name and operating system), the dates and times of your browsing activity and, most importantly, the URLs you visited. Ordinarily, a cryptocurrency address page is only visited by the address owner, while the transaction page is visited by the transaction parties. **Blockchain explorers can therefore easily trace the digital fingerprint that links addresses and transactions. Unfortunately, this data is also picked up by the web analytics tools (Google Analytics, Baidu Tongji, Yandex.Metrica), advertising platforms and similar third-party services.**

User data can be traced in others ways too. CDN providers like Cloudflare, Incapsula and AWS Shield act as reverse proxies, which means some websites require you to request data from a CDN in order to use the site. You therefore share your information with the provider.

In addition to these data tracking services, there are several other ways how users can be identified online.

  * HTTP referer: a client request header that allows a server to trace the previous site you visited. Say you visit example.com followed by explorer.com/1YourBitcoinAddress then the former will receive information that you have come from the latter;
  * Web beacon (bug): an invisible web page element that confirms a user has visited a web page. This is used to collect user analytics;
  * Cookies: user activity data stored in the user’s browser. Third-party cookies can also be embedded in the site’s code (if it contains elements from other sites);
  * Evercookie: a JavaScript app that stores zombie cookies on a computer. These cookies are extremely difficult to remove since Evercookie recreates them whenever they are deleted;
  * Device / browser fingerprint: the device and browser information collected for user identification;
  * Browser extensions.



Most blockchain explorers and cryptocurrency companies store user information, including available balances, lists of transactions and types of cryptocurrency.

They might sell this information, publish it, share it with government agencies, or they might be hacked. If it becomes public knowledge that you have significant funds stored in cryptocurrency, you’re likely to be targeted by cyber criminals. Your personal safety may be at risk too.

  * When you connect to Blockchair your browser automatically sends us information about your computer, User Agent, IP address, and the page you want to visit. Since this data may expose your identity, **we do not permanently store information about you** ;
  * **We do not use cookies that can be used to identify you.** See below for details;
  * **Your browser won’t send HTTP referer headers when leaving Blockchair.com. This means you can move to other sites without your browsing activity being traced by those sites;**
  * **We do not use CDN-providers, including those used to distribute JavaScript libraries and styles. We do not use hit counters, web analytics tools (such as Google Analytics) or any other third-party site elements. Therefore, other parties do not receive information about you.**



We only collect anonymous aggregated data that allows us to improve our website features. We count visitors, analyze popular searches, cryptocurrencies, sortings and other queries.

We also store the incoming IP addresses for short periods of 5 to 10 minutes. This is to limit the rate of API requests.

Your device may store technical cookies, such as those that keep the night mode on. In this case, only the client part of the site interacts with them.

Collected data is used to improve user experience and compile website traffic statistics.

We might activate logging procedure to safeguard our services since we're not protected from certain types of third-party network attacks. If this happens, we will post a notification in the site header to let you know we're collecting additional information during the attack. Once the attack has been stopped, all logs will be deleted along with the notification.

We will publish any updates to our Privacy Policy at this page ([https://blockchair.com/privacy](https://web.archive.org/privacy)) and in the GitHub repository at <https://github.com/Blockchair/Blockchair.Support/blob/master/PRIVACY.md> plus the link to the updated version will be available at the bottom of all our site pages.

Please share your comments and suggestions at <[info@blockchair.com](mailto:info@blockchair.com)>.
