var srzoo = {

	DEFAULT_CONFIG: {
    'channel_first': false,
    'pixel_range': [0.0, 255.0]
	},

	loadConfig: function(url) {
		return new Promise(function(resolve, reject) {
			$.ajax({
				url: url,
				dataType: 'json'
			})
			.done(function(config) {
				for (var key in srzoo.DEFAULT_CONFIG) {
					if (!config.hasOwnProperty(key)) {
						config[key] = srzoo.DEFAULT_CONFIG[key];
					}
				}
				resolve(config);
			})
			.fail(function(jqXHR, textStatus) {
				reject(textStatus);
			})
		});
	},

	processInput: function(config, data) {
		data = data.expandDims(0);

		// adjust channel dimension
		if (config['channel_first']) {
			data = data.transpose([0, 3, 1, 2]);
		}
		
		// adjust pixel range
		data = data.mul(tf.scalar((config['pixel_range'][1] - config['pixel_range'][0]) / 255.0));
		data = data.add(tf.scalar(config['pixel_range'][0]));

		return data;
	},

	processOutput: function(config, data) {
		data = data.clipByValue(0, 1);

		// adjust pixel range to [0, 1]
		data = data.sub(tf.scalar(config['pixel_range'][0]));
		data = data.mul(tf.scalar(1.0 / (config['pixel_range'][1] - config['pixel_range'][0])));

		// adjust channel dimension
		if (config['channel_first']) {
			data = data.transpose([0, 2, 3, 1]);
		}

		// squeeze
		data = data.squeeze([0]);

		return data;
	}

};