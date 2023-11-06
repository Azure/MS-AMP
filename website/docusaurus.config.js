// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'MS-AMP',
  tagline: 'Automatic mixed precision package for deep learning developed by Microsoft',
  url: 'https://azure.github.io',
  baseUrl: '/MS-AMP/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'azure',
  projectName: 'MS-AMP',
  themeConfig: {
    navbar: {
      title: 'MS-AMP',
      logo: {
        alt: 'Docusaurus Logo',
        src: 'img/logo.svg',
      },
      items: [
        // left
        {
          type: 'doc',
          docId: 'introduction',
          label: 'Docs',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        // right
        {
          href: 'https://github.com/azure/MS-AMP',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/docs/introduction',
            },
            {
              label: 'Getting Started',
              to: '/docs/getting-started/installation',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Issues',
              href: 'https://github.com/Azure/MS-AMP/issues',
            },
            {
              label: 'Discussion',
              href: 'https://github.com/Azure/MS-AMP/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/Azure/MS-AMP',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MS-AMP. <br> Built with Docusaurus and hosted by GitHub.`,
    },
    announcementBar: {
      id: 'supportus',
      content:
        '📢 <a href="https://azure.github.io/MS-AMP/blog/release-msamp-v0.3">v0.3.0</a> has been released! ' +
        '⭐️ If you like MS-AMP, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/Azure/MS-AMP">GitHub</a>! ⭐️',
    },
    algolia: {
      apiKey: '6809111d3dabf59fe562601d591d7c53',
      indexName: 'msamp',
      contextualSearch: true,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['ini'],
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: '../docs',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/azure/MS-AMP/edit/main/website/',
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/azure/MS-AMP/edit/main/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/index.css'),
        },
      },
    ],
  ],
};