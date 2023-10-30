// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.


module.exports = {
  docs: [
    'introduction',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/run-msamp',
      ],
    },
    {
      type: 'category',
      label: 'User Tutorial',
      collapsed: false,
      items: [
        'user-tutorial/usage',
        'user-tutorial/optimization-level',
        'user-tutorial/container-images',
      ],
    },
    {
      type: 'category',
      label: 'Developer Guides',
      items: [
        'developer-guides/development',
        'developer-guides/using-docker',
        'developer-guides/contributing',
      ],
    },
  ]
};
